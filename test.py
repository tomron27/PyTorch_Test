from vgg import *

print("Loading model...")

model = vgg11()

print(model)

print("Model parameters: {}".format(sum(p.numel() for p in model.parameters())))
