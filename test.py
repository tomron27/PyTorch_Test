import os
import torch
from vgg import *
from PIL import Image
from torchvision import transforms
from data_loaders import *
from os.path import join
from sklearn.metrics import multilabel_confusion_matrix as MCM
from sklearn.metrics import roc_auc_score
import logging

logger = logging.getLogger(__name__)
handler = logging.FileHandler('test.py.log')
handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', '%Y-%m-%d %H:%M:%S'))
logger.setLevel(logging.INFO)
logger.addHandler(handler)

logger.info("--- test.py Log begin ---")

# Paths
base_dir = "/home/tomron27@st.technion.ac.il/"
project_dir = join(base_dir, "projects/PyTorch_Test/")
data_base_dir = join(base_dir, "projects/ChestXRay/data/fetch/")
images_path = join(data_base_dir, "images/")

val_metadata_path = join(data_base_dir, "validation_metadata.csv")
train_metadata_path = join(data_base_dir, "train_metadata.csv")
test_metadata_path = join(data_base_dir, "test_metadata.csv")

model_dir = join(project_dir, "models/")

# Params
batch_size = 1
input_size = 1024
resize_factor = 2
resize = input_size//resize_factor
tau = 0.3
print_interval = 2000

# Transformations
trans_list = []
if resize_factor > 1:
    trans_list += [transforms.Resize(resize)]

trans_list += [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
trans = transforms.Compose(trans_list)

# Dataloaders
data = ChestXRayDataset(csv_file=test_metadata_path,
                             root_dir=images_path,
                             transform=trans)

data_loader = torch.utils.data.DataLoader(dataset=data,
                                          batch_size=batch_size,
                                          shuffle=False)

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = vgg16_bn(size=512, num_classes=8)
model.to(device)

checkpoint = torch.load(join(model_dir, "20_epochs", "vgg_16_bn_norm_epoch_5.pt"))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Iterate data
y_true_arr = np.zeros(shape=(len(data_loader), len(data.labels_dict)), dtype=np.float64)
y_pred_arr = np.zeros(shape=(len(data_loader), len(data.labels_dict)), dtype=np.float64)

for i, sample in enumerate(data_loader):

    if i % 2000 == 0:
        logger.info("{:05d}...".format(i))

    inputs = sample["image"].to(device)
    labels = sample["label"].to(device)

    outputs = model(inputs)
    sftmax = torch.nn.Softmax()(outputs)

    # sftmax[sftmax >= tau] = 1.0
    # sftmax[sftmax < tau] = 0.0

    y_true_arr[i] = labels.cpu().detach().numpy()[0]
    y_pred_arr[i] = sftmax.cpu().detach().numpy()[0]

# mcm = MCM(y_true_arr, y_pred_arr)
#
# logger.info(mcm)

logger.info("Saving predictions...")
np.save("test_labels.npy", y_true_arr)
np.save("test_pred.npy", y_pred_arr)
