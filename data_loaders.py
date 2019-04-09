import numpy as np
import pandas as pd
import os

## Unpack tar.gz files command:
# for i in {01..09}
# do
#    echo "Extracting images_$i.tar.gz..."
#    tar xvzf images_$i.tar.gz
# done
#
# tar xvzf images_10.tar.gz

base_dir = "/home/tomron27@st.technion.ac.il/"
data_path = base_dir + "projects/ChestXRay/data/fetch/"

data_entry = pd.read_csv(data_path + "Data_Entry_2017.csv")

print(data_entry.head(5))
