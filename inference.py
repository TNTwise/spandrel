import tensorrt
import torch
from libs.spandrel.spandrel.architectures.__arch_helpers.grid_sample_trt_plugin import (
    WarpPluginCreator,
)
from libs.spandrel.spandrel.architectures.SPANPlus.__arch.spanplus import SPANPlus
from trthandler import TorchTensorRTHandler

registry = tensorrt.get_plugin_registry()
registry.register_creator(WarpPluginCreator())
trtHandler = TorchTensorRTHandler()
model = SPANPlus(upsampler="dys")
test = torch.rand(1, 3, 32, 32).cuda()

# inference to fix weight concat issue
model.eval().cuda()

model(test)

trtHandler.build_engine(model, torch.float16, torch.device("cuda"), [test], "trt.ts")
model = trtHandler.load_engine("trt.ts")
model(test)
