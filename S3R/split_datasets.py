import numpy as np
import os
import csv
import pandas as pd
import cv2
from tqdm import tqdm
from multiprocessing import Pool


class SegementationDataset:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        csv_file: str,
        target_shape: int = 32,
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.csv_file = csv_file
        self.output_dir_train = os.path.join(output_dir, "i3d")
        self.target_shape = target_shape
        self.makedirs()

    def makedirs(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.output_dir_train):
            os.makedirs(self.output_dir_train)

    @staticmethod
    def resize_feat(features, target_size):
        features = np.transpose(features, (2, 0, 1))  # C x tau x N
        # quantize each video to 32-snippet-length video
        width, height = target_size, 2048
        features = cv2.resize(features, (width, height), interpolation=cv2.INTER_LINEAR)  # CxTxN
        video = np.transpose(features, (1, 2, 0))  # reset
        return video

    def save_resize_npy(self, file):
        # Load npy file
        file_path = os.path.join(self.input_dir, file + "_i3d.npy")
        data = np.load(file_path)  # Assuming each row in the CSV contains one file path

        # Resize the data
        resize_data = self.resize_feat(data, self.target_shape)
        np.save(os.path.join(self.output_dir_train, f"{file}_i3d.npy"), resize_data)

    def split_npy_files_resize(self):
        # Read the CSV file and get the list of npy file paths
        video_list = pd.read_csv(self.csv_file)
        video_list = video_list["video-id"].values[:]

        p = Pool(24)
        p.map(self.save_resize_npy, video_list)

    def split_npy_files(self):
        # Read the CSV file and get the list of npy file paths
        video_list = pd.read_csv(self.csv_file)
        video_list = video_list["video-id"].values[:]

        normaly_tags = 0
        file_ids = []

        for i, file in enumerate(video_list):
            # Load npy file
            file_path = os.path.join(self.input_dir, file + "_i3d.npy")
            data = np.load(file_path)  # Assuming each row in the CSV contains one file path

            # Calculate the maximum number of complete chunks that can be made
            num_chunks = data.shape[0] // self.target_shape

            # Truncate the data to a size that can be split evenly into chunks
            flag = True
            resize_data = None
            if data.shape[0] % self.target_shape != 0:
                resize_data = data[num_chunks * self.target_shape :]
                flag = False
            data = data[: num_chunks * self.target_shape]

            # Reshape data
            reshaped_data = data.reshape((-1,) + (self.target_shape, 10, 2048))
            if i == 810:
                print("++++++++++++++++++++")
                print(normaly_tags)
                print("++++++++++++++++++++")
            # Save each split array as a new npy file
            if num_chunks != 0:
                for j in range(reshaped_data.shape[0]):
                    file_name = f"{file}-{j+1}"
                    np.save(os.path.join(self.output_dir_train, f"{file_name}_i3d.npy"), reshaped_data[j])
                    file_ids.append(file_name)
                    normaly_tags += 1

            if flag:
                continue
            file_name = f"{file}-{num_chunks+1}"
            print(file_name)
            if padding:
                # Pad the remaining data with zeros
                pad_width = ((0, self.target_shape - resize_data.shape[0]), (0, 0), (0, 0))
                resize_data = np.pad(resize_data, pad_width=pad_width, mode="constant", constant_values=0)
            else:  # resize
                resize_data = self.resize_feat(resize_data)

            np.save(os.path.join(self.output_dir_train, f"{file_name}_i3d.npy"), resize_data)
            file_ids.append(file_name)
            normaly_tags += 1

        # Write the new file paths to a new CSV file
        df = pd.DataFrame(file_ids, columns=["video-id"])
        df.to_csv(self.output_dir + "/" + f"{dataset}-split32.training.csv")


dataset = "ucf-crime"
padding = True

data_file_train = f"data/{dataset}/{dataset}.training.csv"  # video list
# Call the function
input_dir = f"data/{dataset}/i3d/train/"
output_dir = f"data/dataset_resize128_{dataset}"
print(output_dir)
sd = SegementationDataset(input_dir, output_dir, data_file_train, target_shape=128)
sd.split_npy_files_resize()
# sd.split_npy_files(target_shape=32)
# ucf-crime-1
# 23975
# 5830
# 18145

# ucf-crime-2
# 25525
# 6614
# 18911

# original
# 1610
# 800
