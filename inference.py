from libs.spandrel.spandrel.architectures.SPANPlus.__arch.spanplus import SPANPlus
from trthandler import TorchTensorRTHandler

TorchTensorRTHandler()
model = SPANPlus(upsampler="dys")
model.eval().cuda()
