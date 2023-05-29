import numpy as np
import os
import csv
import pandas as pd
import cv2
from tqdm import tqdm

def resize_feat(features, target_size = 32):
    features = np.transpose(features, (2, 0, 1))  # C x tau x N
    # quantize each video to 32-snippet-length video
    width, height = target_size, 2048
    features = cv2.resize(features, (width, height), interpolation=cv2.INTER_LINEAR)  # CxTxN
    video = np.transpose(features, (1, 2, 0))  # reset
    return video


def split_npy_files(csv_file, output_dir, path, target_shape=(32, 10, 2048)):
    # Check if output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_dir_train = os.path.join(output_dir, "i3d")
    if not os.path.exists(output_dir_train):
        os.makedirs(output_dir_train)

    # Read the CSV file and get the list of npy file paths
    video_list = pd.read_csv(csv_file)
    video_list = video_list["video-id"].values[:]

    normaly_tags = 0
    file_ids = []

    for i, file in enumerate(video_list):
        # Load npy file
        file_path = os.path.join(path, file + "_i3d.npy")
        data = np.load(file_path)  # Assuming each row in the CSV contains one file path

        # Calculate the maximum number of complete chunks that can be made
        num_chunks = data.shape[0] // target_shape[0]

        # Truncate the data to a size that can be split evenly into chunks
        flag = True
        resize_data = None
        if data.shape[0] % target_shape[0] != 0:
            resize_data = data[num_chunks * target_shape[0]:]
            flag = False
        data = data[: num_chunks * target_shape[0]]

        # Reshape data
        reshaped_data = data.reshape((-1,) + target_shape)
        if i == 810:
            print("++++++++++++++++++++")
            print(normaly_tags)
            print("++++++++++++++++++++")
        # Save each split array as a new npy file
        if num_chunks != 0:
            for j in range(reshaped_data.shape[0]):
                file_name = f"{file}-{j+1}"
                np.save(os.path.join(output_dir_train, f"{file_name}_i3d.npy"), reshaped_data[j])
                file_ids.append(file_name)
                normaly_tags += 1
        
        if flag:
            continue
        file_name = f"{file}-{num_chunks+1}"
        print(file_name)
        if padding:
            # Pad the remaining data with zeros
            pad_width = ((0, target_shape[0] - resize_data.shape[0]), (0, 0), (0, 0))
            resize_data = np.pad(resize_data, pad_width=pad_width, mode="constant", constant_values=0)
        else: # resize
            resize_data = resize_feat(resize_data)
            
        np.save(os.path.join(output_dir_train, f"{file_name}_i3d.npy"), resize_data)
        file_ids.append(file_name)
        normaly_tags += 1

    # Write the new file paths to a new CSV file
    df = pd.DataFrame(file_ids, columns=["video-id"])
    df.to_csv(output_dir + "/" + f"{dataset}-split32.training.csv")
    

dataset = "ucf-crime"
padding = True

data_file_train = f"data/{dataset}/{dataset}.training.csv"  # video list
# Call the function
split_npy_files(data_file_train, f"data/dataset_split32_{dataset}-p", f"data/{dataset}/i3d/train/")

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
