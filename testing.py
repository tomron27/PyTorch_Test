import os
import torch
from vgg import *
from PIL import Image
from torchvision import transforms

base_dir = "/home/tomron27@st.technion.ac.il/"
project_dir = os.path.join(base_dir, "projects/PyTorch_Test/")
data_base_dir = os.path.join(base_dir, "projects/ChestXRay/data/fetch/")
train_metadata_path = os.path.join(data_base_dir, "train_metadata.csv")
test_metadata_path = os.path.join(data_base_dir, "test_metadata.csv")
images_path = os.path.join(data_base_dir, "images/")

model_dir = os.path.join(project_dir, "models/")

image_path_1 = os.path.join(images_path, "00007245_000.png")
image_path_2 = os.path.join(images_path, "00001346_000.png")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trans_list = [transforms.Resize(512), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
trans = transforms.Compose(trans_list)

image_1 = trans(Image.open(image_path_1).convert('L')).to(device).unsqueeze(0)
image_2 = trans(Image.open(image_path_2).convert('L')).to(device).unsqueeze(0)

model = vgg16_bn(size=512, num_classes=8)
model.to(device)

checkpoint = torch.load(os.path.join(model_dir, "vgg_16_bn_epoch_1.pt"))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

image_1_pred = model(image_1)
image_2_pred = model(image_2)

print(image_1_pred)
print(image_2_pred)

pass