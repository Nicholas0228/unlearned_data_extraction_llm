import os
import json
import numpy as np
from matplotlib import pyplot as plt
# Define the keywords and the directory to search
key_words = ['rougeL_recall']
src_directory = '../checkpoint_updated/MUSE'  # Current directory

threshold_list = [0.9, 0.99, 0.0]
use_threshold_list = [True, True, False]


# Initialize a dictionary to store values for each keyword

for threshold, use_threshold in zip(threshold_list, use_threshold_list):
    if use_threshold:
        print(f'Metric: A-ESR with Threshold {threshold}')
    else:
        print('Metric: Average RougeL(R)')
    for dirname in os.listdir(src_directory):
        basedirname = os.path.join(src_directory, dirname)

        for checkpointname in os.listdir(basedirname):
            if not checkpointname.startswith('checkpoint-5553'):
                continue
            directory = os.path.join(basedirname, checkpointname)
            # Traverse all JSON files in the directory
            for key in key_words:
                for filename in sorted(os.listdir(directory)):
                    if filename.endswith('.json'):
                        if 'forget' not in filename or 'False_5.0' not in filename:
                            continue
                        file_path = os.path.join(directory, filename)
                        # Open and read each JSON file
                        with open(file_path, 'r') as f:
                            data = json.load(f)

                            # Check if the key exists and is a list
                            if key in data and isinstance(data[key], list):

                                if use_threshold:
                                    input_data = [v>threshold for v in data[key]]
                                else:
                                    input_data = [v for v in data[key]]
                
                                mean_value = np.mean(input_data)
                            else:
                                print(f"No valid data for {key} in file {filename}")
                        if 'True_1.0' in filename:
                            print(f"Post-Unlearning Extraction, {mean_value}")
                        elif 'True_-1.0' in filename:
                            print(f"Pre-Unlearning Extraction, {mean_value}")
                        elif 'True_-2.0' in filename:
                            print(f"Our Extraction, {mean_value}")

