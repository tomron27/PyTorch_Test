import pandas as pd

all_data = pd.read_csv("Data_Entry_2017.csv")
train = pd.read_csv("train_val_list.txt")
train.columns = ["Image Index"]
test = pd.read_csv("test_list.txt")
test.columns = ["Image Index"]

all_data.merge(train, on="Image Index").to_csv("train_metadata.csv", index=False)
all_data.merge(test, on="Image Index").to_csv("test_metadata.csv", index=False)