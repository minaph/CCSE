
raise NotImplementedError

from detectron2.engine import DefaultPredictor
from common.cmd_parser import parse_cmd_arg
from initializer.instance_initializer import InferenceInstanceInitializer

from detectron2.export import Caffe2Tracer

import torch
import onnx
import os


predictor: DefaultPredictor = None
dataset_metadata = None


def model_init(init: InferenceInstanceInitializer):
    global dataset_metadata, predictor
    config = init.config
    dataset_metadata = init.dataset_metadata

    predictor = DefaultPredictor(config)
    dummy_input = torch.randn(3, 120, 120)

    tracer = Caffe2Tracer(config, predictor, dummy_input)

    # ONNX形式に変換する
    # torch.onnx.export(predictor, dummy_input, "model.onnx", input_names=["input"], output_names=["output"])

    # ONNXモデルをファイルに保存する
    # onnx.save_model(onnx_model, "model.onnx")

    
    onnx_model = tracer.export_onnx()
    onnx.save(onnx_model, "model.onnx")

def export_tracing(torch_model, inputs):
    assert TORCH_VERSION >= (1, 8)
    image = inputs[0]["image"]
    inputs = [{"image": image}]  # remove other unused keys

    if isinstance(torch_model, GeneralizedRCNN):

        def inference(model, inputs):
            # use do_postprocess=False so it returns ROI mask
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

    else:
        inference = None  # assume that we just call the model directly

    traceable_model = TracingAdapter(torch_model, inputs, inference)

    if args.format == "torchscript":
        ts_model = torch.jit.trace(traceable_model, (image,))
        with PathManager.open(os.path.join(args.output, "model.ts"), "wb") as f:
            torch.jit.save(ts_model, f)
        dump_torchscript_IR(ts_model, args.output)
    elif args.format == "onnx":
        with PathManager.open(os.path.join(args.output, "model.onnx"), "wb") as f:
            torch.onnx.export(traceable_model, (image,), f, opset_version=STABLE_ONNX_OPSET_VERSION)
    logger.info("Inputs schema: " + str(traceable_model.inputs_schema))
    logger.info("Outputs schema: " + str(traceable_model.outputs_schema))

    if args.format != "torchscript":
        return None
    if not isinstance(torch_model, (GeneralizedRCNN, RetinaNet)):
        return None

    def eval_wrapper(inputs):
        """
        The exported model does not contain the final resize step, which is typically
        unused in deployment but needed for evaluation. We add it manually here.
        """
        input = inputs[0]
        instances = traceable_model.outputs_schema(ts_model(input["image"]))[0]["instances"]
        postprocessed = detector_postprocess(instances, input["height"], input["width"])
        return [{"instances": postprocessed}]

    return eval_wrapper




if __name__ == '__main__':
    args = parse_cmd_arg()

    initializer = InferenceInstanceInitializer(args.config)
    model_init(initializer)