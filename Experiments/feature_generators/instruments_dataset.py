import csv
import os
import shutil

import pandas as pd
from pydub import AudioSegment
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

DATASET_PATH = "../wavfiles/instruments"
NEW_DATASET_PATH = "E:\\LICENTA\\instruments_equalised"
OUT_PATH = "../statistics/instruments_complete_features.csv"
CSV_PATH = "../statistics/instruments.csv"
SAMPLE_RATE = 22050
TRACK_DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

FEATURE_NAMES = ["chroma_stft", "rms", "spectral_centroid", "spectral_bandwidth", "rolloff", "zero_crossing_rate", "harmony"]

def preprocess_audio(file_path, new_file_path, target_length = 22050*3):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Check the duration of the audio file
    duration = len(audio)

    if duration > target_length:
        # Trim the audio to the target length
        audio = audio[:target_length]
    elif duration < target_length:
        # Calculate the amount of silence needed
        silence_duration = target_length - duration
        # Create a silence audio segment
        silence = np.zeros(silence_duration, dtype=audio.dtype)
        # Concatenate audio with silence at the end
        audio = np.concatenate((audio, silence))
        # print(audio.shape)

    wavfile.write(new_file_path, SAMPLE_RATE, audio)

def create_equalised_dataset(dataset_path, new_dataset_path, csv_path):
    df = pd.read_csv(csv_path)
    df.set_index('fname', inplace=True)
    classes = list(np.unique(df.label))
    for class_name in classes:
        class_path = os.path.join(new_dataset_path, class_name)
        os.makedirs(class_path)
        filtered_df = df[df["label"] == class_name]
        for f in filtered_df.index:
            old_path = os.path.join(dataset_path, f)
            preprocess_audio(old_path, os.path.join(class_path, f))

def extract_features_from_data(dataset_path, output_path):
    data = [["filename", "length"]]
    for name in FEATURE_NAMES:
        data[0].append(name + "_mean")
        data[0].append(name + "_var")
    data[0].append("tempo")
    for i in range(20):
        data[0].append("mfcc" + str(i+1)+"_mean")
        data[0].append("mfcc" + str(i+1)+"_var")

    data[0].append("label")

    for folder_name in os.listdir(dataset_path):
        print("Started ", folder_name)
        if folder_name != "voi":
            class_path = os.path.join(dataset_path, folder_name)
            for file_name in tqdm(os.listdir(class_path)):
                abs_file_name = os.path.join(class_path, file_name)
                audio, sr = librosa.load(abs_file_name, sr=SAMPLE_RATE)
                data_row = [abs_file_name, len(audio)]

                chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
                rms = librosa.feature.rms(y=audio)
                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
                spectral_bandwidths = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
                zero_crossing_rates = librosa.feature.zero_crossing_rate(y=audio)
                chroma_cens = librosa.feature.chroma_cens(y=audio, sr=sr)
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
                mfccs = librosa.feature.mfcc(y=audio, sr=sr)

                data_row.append(np.mean(chroma_stft))
                data_row.append(np.var(chroma_stft))
                data_row.append(np.mean(rms))
                data_row.append(np.var(rms))
                data_row.append(np.mean(spectral_centroids))
                data_row.append(np.var(spectral_centroids))
                data_row.append(np.mean(spectral_bandwidths))
                data_row.append(np.var(spectral_bandwidths))
                data_row.append(np.mean(spectral_rolloff))
                data_row.append(np.var(spectral_rolloff))
                data_row.append(np.mean(zero_crossing_rates))
                data_row.append(np.var(zero_crossing_rates))
                data_row.append(np.mean(chroma_cens))
                data_row.append(np.var(chroma_cens))
                data_row.append(tempo)
                for i, mfcc_column in zip(range(20), mfccs):
                    data_row.append(np.mean(mfcc_column))
                    data_row.append(np.var(mfcc_column))
                data_row.append(folder_name)

                data.append(data_row)
        print("Ended ", folder_name)

    with open(output_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

# create_equalised_dataset(DATASET_PATH, NEW_DATASET_PATH, CSV_PATH)
extract_features_from_data(NEW_DATASET_PATH, OUT_PATH)