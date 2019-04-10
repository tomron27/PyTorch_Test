import shutil
import os
import matplotlib.pyplot as plt


# print('Image name: {}'.format(img_name))
# print('Label string: {}'.format(label_string))
# # print('Labels: {}'.format(parsed_labels))
#
# plt.figure()
# image_path = os.path.join(images_path, img_name)
# image = io.imread(image_path, as_grey=True)
# plt.imsave(os.path.join(base_dir, "fig.png"), image, cmap='gray')
# # plt.savefig(os.path.join(base_dir, "fig.png"))


## Unpack tar.gz files command:
# for i in 01 02 03 04 05 06 07 08 09 10 11 12
# do
#    echo "Extracting images_$i.tar.gz..."
#    tar xvzf images_$i.tar.gz
# done
#
# tar xvzf images_10.tar.gz


def plot_sample(data, size, output_dir):

    for i in range(size):
        sample = data[i]

        print(i, sample["image_name"], sample["labels_str"])

        ax = plt.subplot(size, 1, i + 1)
        # plt.tight_layout()
        ax.set_title('{}, {}'.format(sample["image_name"], sample["labels_str"]), fontsize=8)
        ax.axis('off')
        plt.imshow(sample["image"])

    plt.savefig(os.path.join(output_dir, "fig.png"))


# def split_train_test(source):
#
#     # source = "/home/tomron27@st.technion.ac.il/projects/ChestXRay/data/fetch/"
#     train_metadata_path = os.path.join(source, "train_val_list.txt")
#     test_metadata_path = os.path.join(source, "test_list.txt")
#
#     train_set = set(line.strip() for line in open(train_metadata_path))
#     test_set = set(line.strip() for line in open(test_metadata_path))
#
#     train_dest_path = os.path.join(source, "train")
#     test_dest_path = os.path.join(source, "test")
#
#     if not os.path.exists(train_dest_path):
#         os.mkdir(train_dest_path)
#     if not os.path.exists(test_dest_path):
#         os.mkdir(test_dest_path)
#
#     train_count = 0
#     for file in train_set:
#         try:
#             shutil.copy2(os.path.join(source, "images", file), train_dest_path)
#             train_count += 1
#         except:
#             pass
#
#         print("Copied {} out of {} train images".format(train_count, len(train_set)))
#
#     test_count = 0
#     for file in test_set:
#         try:
#             shutil.copy2(os.path.join(source, "images", file), test_dest_path)
#             test_count += 1
#         except:
#             pass
#
#         print("Copied {} out of {} test images".format(test_count, len(test_set)))
