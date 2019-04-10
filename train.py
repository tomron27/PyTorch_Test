from vgg import *
from data_loaders import *


# Paths
base_dir = "/home/tomron27@st.technion.ac.il/"
data_base_dir = os.path.join(base_dir, "projects/ChestXRay/data/fetch/")
train_metadata_path = os.path.join(data_base_dir, "train_metadata.csv")
test_metadata_path = os.path.join(data_base_dir, "test_metadata.csv")
images_path = os.path.join(data_base_dir, "images/")

# Hyper Parameters
batch_size = 10

# Data loaders
train_data = ChestXRayDataset(csv_file=train_metadata_path,
                             root_dir=images_path,
                             transform=transforms.Compose([transforms.ToTensor()]))

test_data = ChestXRayDataset(csv_file=test_metadata_path,
                             root_dir=images_path,
                             transform=transforms.Compose([transforms.ToTensor()]))


train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size,
                                          shuffle=True)
# Train procedure
for i, sample in enumerate(train_loader):
    print(i, sample["label_str"])
    break
