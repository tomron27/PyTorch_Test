from vgg import *
import torch
from torchvision import transforms
from torch.utils.data import Subset
from data_loaders import *
import time
import json


# Data paths
base_dir = "/home/tomron27@st.technion.ac.il/"
project_dir = os.path.join(base_dir, "projects/PyTorch_Test/")
data_base_dir = os.path.join(base_dir, "projects/ChestXRay/data/fetch/")
train_metadata_path = os.path.join(data_base_dir, "train_metadata.csv")
test_metadata_path = os.path.join(data_base_dir, "test_metadata.csv")
images_path = os.path.join(data_base_dir, "images/")

model_dir = os.path.join(project_dir, "models/")

# Hyper Parameters
num_classes = 8
batch_size = 1
num_epochs = 10
print_interval = 2000
input_size = 1024
resize_factor = 2
resize = input_size//resize_factor
subset = False
subset_size = 2000

trans_list = []
if resize_factor > 1:
    trans_list += [transforms.Resize(resize)]

trans_list += [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
trans = transforms.Compose(trans_list)

# Data loaders
train_data = ChestXRayDataset(csv_file=train_metadata_path,
                             root_dir=images_path,
                             transform=trans)

test_data = ChestXRayDataset(csv_file=test_metadata_path,
                             root_dir=images_path,
                             transform=trans)

# For testing purposes
if subset:
    test_data = Subset(test_data, range(subset_size))
    train_data = Subset(train_data, range(subset_size))

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size,
                                          shuffle=True)

model = vgg16_bn(size=resize, num_classes=num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# print(model)
# num_params = sum((p.numel() for p in model.parameters()))
# print(num_params)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

criterion = torch.nn.BCEWithLogitsLoss()

# Train procedure
start = time.time()
loss_list = []
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)

    batch_train_running_loss = 0.0
    epoch_train_running_loss = 0.0
    batch_test_running_loss = 0.0
    epoch_test_running_loss = 0.0

    model.train()
    for i, sample in enumerate(train_loader):

        inputs = sample["image"].to(device)
        labels = sample["label"].to(device)

        try:
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs.double(), labels)

            # Backward
            loss.backward()
            optimizer.step()

            # log and print batch statistics
            batch_train_running_loss += loss.item()
            epoch_train_running_loss += loss.item()
            if i % print_interval == print_interval - 1:
                print('Train: [%d, %5d] loss: %.3f, %.2f mins' %
                      (epoch + 1, (i+1)*batch_size, batch_train_running_loss / print_interval*batch_size, (time.time()-start)/60))
                batch_train_running_loss = 0.0

                # assertion
                # with torch.no_grad():
                #     print(sample["image_name"], sample["label"], outputs)

        except Exception as e:
            print("Error on training:", e)
            print(sample["image_name"])

    loss_list.append(('train', 'epoch_{}'.format(epoch+1), epoch_train_running_loss / len(train_loader), (time.time()-start)/60))
    epoch_train_running_loss = 0.0

    # Test Procedure
    model.eval()
    for i, sample in enumerate(test_loader):
        inputs = sample["image"].to(device)
        labels = sample["label"].to(device)

        try:
            outputs = model(inputs)
            loss = criterion(outputs.double(), labels)

            # print batch statistics
            batch_test_running_loss += loss.item()
            epoch_test_running_loss += loss.item()
            if i % print_interval == print_interval - 1:
                print('Test:  [%d, %5d] loss: %.3f, %.2f mins' %
                      (epoch + 1, (i+1)*batch_size, batch_test_running_loss / print_interval*batch_size, (time.time()-start)/60))
                batch_test_running_loss = 0.0

        except Exception as e:
            print("Error on testing:", e)
            print(sample["image_name"])

    loss_list.append(('test', 'epoch_{}'.format(epoch+1), epoch_test_running_loss / len(test_loader), (time.time()-start)/60))
    epoch_test_running_loss = 0.0

    # Save Model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(model_dir, "vgg_16_bn_norm_epoch_{}.pt".format(epoch + 1)))

# Log loss results
with open('loss_log.json', 'w') as out:
    json.dump(loss_list, out)


