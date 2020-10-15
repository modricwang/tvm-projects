import time

import cv2
import numpy as np
import onnx
import tvm
import tvm.contrib.graph_runtime as runtime
import tvm.relay as relay
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

import os


def tune_tasks(
        tasks,
        measure_option,
        tuner="xgb",
        n_trial=30,
        early_stopping=None,
        log_filename="tuning.log",
        use_transfer_learning=True):
    # create tmp log file
    import os
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    # os.remove(tmp_log_file)


if __name__ == '__main__':
    # load image
    image_path = '9331584514251_.pic_hd.jpg'
    image = cv2.imread(image_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pre_start = time.time()

    resize_shape = (300, 300)
    img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
    mean = np.array([123.675, 116.28, 103.53]).reshape(1, -1)
    std = np.array([1., 1., 1.]).reshape(1, -1)
    img = img.astype(np.float32)
    img = cv2.subtract(img, mean)
    img = cv2.multiply(img, std)
    img = img.transpose(2, 0, 1)
    pre_end = time.time()

    # port = 9
    # remote = rpc.connect("192.168.50.86", port)
    # remote.upload('./model_tuned/deploy_lib.tar')
    # rlib = remote.load_module("deploy_lib.tar")
    # print(remote)

    # load onnx model and build tvm runtime
    # target = 'llvm  -device=arm_cpu -target='
    target = tvm.target.target.arm_cpu("rasp4b64")
    print(str(target))
    # target = "llvm -keys=arm_cpu -device=arm_cpu -mattr=+neon -mcpu=cortex-a72 -model=bcm2711 -mtriple=aarch64-linux-gnu"

    # ctx = tvm.context(str(target))
    # ctx = remote.cpu()
    dtype = 'float32'
    mssd = onnx.load('mssd.onnx')

    input_blob = mssd.graph.input[0]
    input_shape = tuple(map(lambda x: getattr(x, 'dim_value'), input_blob.type.tensor_type.shape.dim))
    shape_dict = {input_blob.name: input_shape}
    mod, params = relay.frontend.from_onnx(mssd, shape_dict)
    p = 1

    from tvm import autotvm

    tuning_option = {
        "log_filename": './autotune.txt',
        "tuner": "random",
        "n_trial": 100,
        "early_stopping": 80,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.RPCRunner(
                "rasp4b64",
                host="0.0.0.0",
                n_parallel=16,
                port=9190,
                number=1,
                timeout=10
            ),
        ),
    }
    tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

    # p = 1

    tune_tasks(tasks, **tuning_option)
    #
    # measure_option = autotvm.measure_option(
    #     builder=autotvm.LocalBuilder(),
    #     runner=autotvm.RPCRunner(key='arm_cpu', host="192.168.50.86", port=9090)
    # )
    # tuner = autotvm.tuner.XGBTuner(task)
    # tuner.tune(n_trial=20,
    #            measure_option=measure_option,
    #            callbacks=[autotvm.callback.log_to_file('RPC_Tune.log')])

    with relay.build_config(opt_level=3):
        # print(remote)
        graph, lib, params = relay.build(mod, target, target_host=target, params=params)
        # exit(0)
        # tvm.contrib.graph_runtime.GraphModule
    # ######## export lib ########
    path = 'model_tuned/'
    import os

    os.makedirs(path, exist_ok=True)
    path_lib = path + "deploy_lib.tar"
    path_graph = path + "deploy_graph.json"
    path_params = path + "deploy_param.params"
    lib.export_library(path_lib)
    with open(path_graph, "w") as fo:
        fo.write(graph)
    with open(path_params, "wb") as fo:
        fo.write(relay.save_param_dict(params))

    exit(0)

    ######## load lib ########
    # load the module back.
    # graph = open(path_graph).read()
    # lib = tvm.runtime.load_module(path_lib)
    # params = bytearray(open(path_params, "rb").read())

    module = runtime.create(graph, lib, ctx)

    # run
    # module.load_params(params)
    module.set_input(**params)
    module.set_input('input.1', tvm.nd.array(img))

    start = time.time()
    module.run()
    end = time.time()

    # generate anchor
    from anchor import gen_anchors

    mlvl_anchors = gen_anchors()

    img_shape = image.shape
    scale_factor = [img_shape[1] / resize_shape[1], img_shape[0] / resize_shape[0]]  # x_scale, y_scale

    from bbox_utils import get_bboxes_single
    from easydict import EasyDict

    cfg = dict(
        nms=dict(type='nms', iou_thr=0.45),
        min_bbox_size=0,
        score_thr=0.6,
        max_per_img=200
    )

    cfg = EasyDict(cfg)

    # get output
    post_start = time.time()
    cls_score_list = [module.get_output(i).asnumpy()[0] for i in range(6)]
    bbox_pred_list = [module.get_output(i + 6).asnumpy()[0] for i in range(6)]

    # recover bbox
    proposals = get_bboxes_single(cls_score_list, bbox_pred_list, mlvl_anchors, resize_shape, scale_factor, cfg,
                                  rescale=True)
    post_end = time.time()

    from vis_bbox import imshow_det_bboxes

    bboxes = proposals[0]
    labels = proposals[1]
    imshow_det_bboxes(image, bboxes, labels, score_thr=0.9, out_file='out.png')

    print("pre: {}".format(pre_end - pre_start))
    print("run: {}".format(end - start))
    print("post: {}".format(post_end - post_start))
