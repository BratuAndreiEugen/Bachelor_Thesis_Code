import json

import librosa
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import shutil

from tqdm import tqdm

SAMPLE_RATE = 22050
TRACK_DURATION = 3  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def empty_subdirectories(parent_directory):
    for subdir in os.listdir(parent_directory):
        subdir_path = os.path.join(parent_directory, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove the file
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove the directory and its contents
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

empty_subdirectories("gtzan_mel_specs")

with open("../config.json") as json_data_file:
    general_config = json.load(json_data_file)

dataset_path = general_config["GTZAN_Path_Alt"]

df = pd.read_csv('../statistics/gtzan_features_3_sec.csv')

samples_per_segment = int(SAMPLES_PER_TRACK / 10)

for index, row in tqdm(df.iterrows()):
    if row["filename"].split(".")[-2] == "0":
        file_path = dataset_path + "\\" + row["label"] + "\\" + row["filename"].replace(".", "_", 1).split(".")[0]+".wav"

        try:
            # Assuming you have your audio signal `signal` and its sample rate `rate`
            signal, rate = librosa.load(file_path, sr=22050)
        except:
            continue
        for d in range(10):

            # calculate start and finish sample for current segment
            start = samples_per_segment * d
            finish = start + samples_per_segment

            # # extract mfcc
            # mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
            #                             hop_length=hop_length)
            # mfcc = mfcc.T
            #
            # # store only mfcc feature with expected number of vectors
            # if len(mfcc) == num_mfcc_vectors_per_segment:
            #     data["mfcc"].append(mfcc.tolist())
            #     data["labels"].append(i - 1)
            #     print("{}, segment:{}".format(file_path, d + 1))

            # Compute the Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=signal[start:finish], sr=rate)

            # Convert to log scale (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            # Plot the Mel spectrogram
            fig, ax = plt.subplots(figsize=(1.28, 1.3), dpi=100)  # Adjust figure size to 128x130 pixels

            # Display the spectrogram with no axis for cleaner image
            librosa.display.specshow(mel_spec_db, sr=rate, x_axis='time', y_axis='mel', fmax=8000, ax=ax)
            plt.axis('off')  # Turn off the axis

            # Adjust layout to remove any padding/margins
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

            # Save the figure as a JPG file with the exact dimensions
            relative_file_path = row["filename"].replace(".", "_", 1).split(".")[0] + f"_{d}.jpg"
            output_file = f'gtzan_mel_specs//{row["label"]}//{relative_file_path}'
            plt.savefig(output_file, format='jpg', bbox_inches='tight', pad_inches=0)
            plt.close()  # Close the plot to free up memory

