from vgg import *
from torch import nn, optim
from torchvision import transforms
from data_loaders import *


# Data paths
base_dir = "/home/tomron27@st.technion.ac.il/"
data_base_dir = os.path.join(base_dir, "projects/ChestXRay/data/fetch/")
train_metadata_path = os.path.join(data_base_dir, "train_metadata.csv")
test_metadata_path = os.path.join(data_base_dir, "test_metadata.csv")
images_path = os.path.join(data_base_dir, "images/")

# Hyper Parameters
num_classes = 8
batch_size = 8
num_epochs = 1
resize_factor = 2
input_size = 1024
resize = input_size//resize_factor

trans_list = []
if resize_factor > 1:
    trans_list += [transforms.Resize(resize)]

trans_list += [transforms.ToTensor()]
trans = transforms.Compose(trans_list)

# Data loaders
train_data = ChestXRayDataset(csv_file=train_metadata_path,
                             root_dir=images_path,
                             transform=trans)

test_data = ChestXRayDataset(csv_file=test_metadata_path,
                             root_dir=images_path,
                             transform=trans)


train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size,
                                          shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = vgg11(size=resize, num_classes=num_classes)

if torch.cuda.is_available():
    model.cuda()

# print(model)
# num_params = sum((p.numel() for p in model.parameters()))
# print(num_params)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

criterion = nn.BCEWithLogitsLoss()

# Train procedure

model.train()

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    for i, sample in enumerate(train_loader):
        inputs = sample["image"].to(device)
        labels = sample["label"].to(device)

        optimizer.zero_grad()

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs.double(), labels)

        # Backward
        loss.backward()
        optimizer.step()
