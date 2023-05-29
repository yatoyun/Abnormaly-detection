import numpy as np
import os
import csv

def split_npy_files(csv_file, output_dir, target_shape=(32,10,2048)):
    # Check if output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the CSV file and get the list of npy file paths
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        npy_files = list(reader)

    for file in npy_files:
        # Load npy file
        data = np.load(file[0])  # Assuming each row in the CSV contains one file path

        # Calculate the maximum number of complete chunks that can be made
        num_chunks = data.shape[0] // target_shape[0]

        # If no complete chunks can be made, skip the file
        if num_chunks == 0:
            print(f"Cannot split {file[0]} into shape {target_shape}. Skipping file.")
            continue

        # Truncate the data to a size that can be split evenly into chunks
        data = data[:num_chunks * target_shape[0]]

        # Reshape data
        reshaped_data = data.reshape((-1,) + target_shape)

        # Save each split array as a new npy file
        for i in range(reshaped_data.shape[0]):
            np.save(os.path.join(output_dir, f"{os.path.basename(file[0]).split('.')[0]}_part{i}.npy"), reshaped_data[i])

# Call the function
split_npy_files('file_paths.csv', 'output_directory')
