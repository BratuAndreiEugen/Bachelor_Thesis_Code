import csv
import os
import shutil

from pydub import AudioSegment
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

DATASET_PATH = "E:\\LICENTA\\Philarmonia"
NEW_DATASET_PATH = "E:\\LICENTA\\Philarmonia_equalised"
OUT_PATH = "../statistics/philarmonia_complete_features.csv"
SAMPLE_RATE = 22050
TRACK_DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

FEATURE_NAMES = ["chroma_stft", "rms", "spectral_centroid", "spectral_bandwidth", "rolloff", "zero_crossing_rate",
                 "harmony"]


def preprocess_audio(file_path, new_file_path,
                     target_length=3, sample_rate=22050):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    target_length = target_length * 22050
    duration = len(audio)

    if duration > target_length:
        audio = audio[:target_length]
    elif duration < target_length:
        silence_duration = target_length - duration
        silence = np.zeros(silence_duration, dtype=audio.dtype)
        audio = np.concatenate((audio, silence))

    wavfile.write(new_file_path, SAMPLE_RATE, audio)


def create_equalised_dataset(dataset_path, new_dataset_path):
    for folder_name in os.listdir(new_dataset_path):
        directory = os.path.join(new_dataset_path, folder_name)
        # Remove the directory with all its content
        shutil.rmtree(directory)
        # Optionally, recreate the directory
        os.makedirs(directory)

    for folder_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, folder_name)
        new_class_path = os.path.join(new_dataset_path, folder_name)
        print(class_path)
        print(new_class_path)
        if folder_name == "percussion":
            for subfolder_name in os.listdir(class_path):
                subfolder_path = os.path.join(class_path, subfolder_name)
                for file_name in os.listdir(subfolder_path):
                    abs_file_name = os.path.join(subfolder_path, file_name)
                    new_abs_file_name = os.path.join(new_class_path, file_name)
                    preprocess_audio(abs_file_name, new_abs_file_name)
            continue

        for file_name in os.listdir(class_path):
            abs_file_name = os.path.join(class_path, file_name)
            new_abs_file_name = os.path.join(new_class_path, file_name)
            preprocess_audio(abs_file_name, new_abs_file_name)


def extract_features_from_data(dataset_path, output_path):
    data = [["filename", "length"]]
    for name in FEATURE_NAMES:
        data[0].append(name + "_mean")
        data[0].append(name + "_var")
    data[0].append("tempo")
    for i in range(20):
        data[0].append("mfcc" + str(i + 1) + "_mean")
        data[0].append("mfcc" + str(i + 1) + "_var")

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


# create_equalised_dataset(DATASET_PATH, NEW_DATASET_PATH)
extract_features_from_data(NEW_DATASET_PATH, OUT_PATH)
