import torch
from vgg import *

print(torch.__version__)

model = vgg16_bn()

print(model)
