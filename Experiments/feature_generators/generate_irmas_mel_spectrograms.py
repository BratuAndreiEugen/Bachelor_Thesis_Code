import json

import librosa
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

with open("../config.json") as json_data_file:
    general_config = json.load(json_data_file)

dataset_path = general_config["IRMAS_Training_Path_Alt"]

df = pd.read_csv('../statistics/irmas_novoice.csv')

for index, row in df.iterrows():
    file_path = dataset_path + "//" + row["filename"]

    signal, rate = librosa.load(file_path, sr=22050)

    mel_spec = librosa.feature.melspectrogram(y=signal, sr=rate)
    print(mel_spec.shape)

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    print(mel_spec_db.shape)
    fig, ax = plt.subplots(figsize=(1.28, 1.3), dpi=100)

    librosa.display.specshow(mel_spec_db, sr=rate, x_axis='time', y_axis='mel', fmax=8000, ax=ax)
    plt.axis('off')  # Turn off the axis

    # Adjust layout to remove any padding/margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Save the figure as a JPG file with the exact dimensions
    output_file = f'irmas_mel_specs//{row["filename"].split(".")[0]+".jpg"}'
    plt.savefig(output_file, format='jpg', bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to free up memory

    print(f'Mel spectrogram saved as {output_file}')
