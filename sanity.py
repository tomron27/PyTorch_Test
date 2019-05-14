import os
import torch
from vgg import *
from PIL import Image
from torchvision import transforms
from data_loaders import *
from os.path import join

# Paths
base_dir = "/home/tomron27@st.technion.ac.il/"
project_dir = join(base_dir, "projects/PyTorch_Test/")
data_base_dir = join(base_dir, "projects/ChestXRay/data/fetch/")
val_dir = join(data_base_dir, "validation_small")
val_metadata_path = join(val_dir, "val_small_metadata.csv")
train_metadata_path = join(data_base_dir, "train_metadata.csv")
test_metadata_path = join(data_base_dir, "test_metadata.csv")
images_path = join(data_base_dir, "images/")

model_dir = join(project_dir, "models/")

# Params
batch_size = 1
input_size = 1024
resize_factor = 2
resize = input_size//resize_factor

# Transformations
trans_list = []
if resize_factor > 1:
    trans_list += [transforms.Resize(resize)]

trans_list += [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
trans = transforms.Compose(trans_list)

# Dataloaders
val_data = ChestXRayDataset(csv_file=val_metadata_path,
                             root_dir=val_dir,
                             transform=trans)

val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                          batch_size=batch_size,
                                          shuffle=False)

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = vgg16_bn(size=512, num_classes=8)
model.to(device)

checkpoint = torch.load(join(model_dir, "vgg_16_bn_norm_epoch_10.pt"))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Iterate data
for i, sample in enumerate(val_loader):

    inputs = sample["image"].to(device)
    labels = sample["label"].to(device)

    outputs = model(inputs)
    sftmax = torch.nn.Softmax()(outputs)

    y_true = labels.cpu().detach().numpy()
    y_pred = sftmax.cpu().detach().numpy()

    # arr[arr > tau] = 1.0

    pass