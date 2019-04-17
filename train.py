from vgg import *
import torch
from torchvision import transforms
from data_loaders import *
import time


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
batch_size = 2
num_epochs = 1
print_interval = 1000
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

model = vgg16_bn(size=resize, num_classes=num_classes)

if torch.cuda.is_available():
    model.cuda()

# print(model)
# num_params = sum((p.numel() for p in model.parameters()))
# print(num_params)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

criterion = torch.nn.BCEWithLogitsLoss()

# Train procedure

model.train()

start = time.time()

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    running_loss = 0.0
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

            # print statistics
            running_loss += loss.item()
            if i % print_interval == print_interval - 1:
                print('[%d, %5d] loss: %.3f, %.2f mins' %
                      (epoch + 1, (i+1)*batch_size, running_loss / print_interval*batch_size, (time.time()-start)/60))
                running_loss = 0.0

                # assertion
                # with torch.no_grad():
                #     print(sample["image_name"], sample["label"], outputs)

        except Exception as e:
            print(e)
            print(sample["image_name"])

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(model_dir, "vgg_16_bn_epoch_{}.pt".format(epoch + 1)))
