from __future__ import print_function, division
import torch
import pandas as pd
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms
from data_utils import *


base_dir = "/home/tomron27@st.technion.ac.il/"
data_base_dir = os.path.join(base_dir, "projects/ChestXRay/data/fetch/")
train_metadata_path = os.path.join(data_base_dir, "train_metadata.csv")
test_metadata_path = os.path.join(data_base_dir, "test_metadata.csv")
images_path = os.path.join(data_base_dir, "images/")


class ChestXRayDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.labels_dict = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3,
               "Mass": 4, "Nodule": 5, "Normal": 6, "Pneumonia": 7, "Pneumothorax": 8}

    def get_label_array(self, label_string):
        label_list = label_string.split("|")
        result = np.zeros(len(self.labels_dict))
        for label in label_list:
            if label in self.labels_dict:
                result[self.labels_dict[label]] += 1
        return result

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        img_name = self.metadata_df.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, img_name)
        image = io.imread(img_path, as_grey=True)
        label_str = self.metadata_df.iloc[idx, 1]
        label_arr = self.get_label_array(label_str)

        sample = {"image": image, "label": label_arr, "label_str": label_str, "image_name": img_name}

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample


train_data = ChestXRayDataset(csv_file=train_metadata_path,
                             root_dir=images_path,
                             transform=transforms.Compose([transforms.ToTensor()]))

test_data = ChestXRayDataset(csv_file=test_metadata_path,
                             root_dir=images_path,
                             transform=transforms.Compose([transforms.ToTensor()]))


train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                          batch_size=10,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=10,
                                          shuffle=True)

for i, sample in enumerate(train_loader):
    print(i, sample["label_str"])
    break
    # split_train_test(data_base_dir)

