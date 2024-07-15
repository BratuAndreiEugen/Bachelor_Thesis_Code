import csv
import os

import librosa
import numpy as np

DATASET_PATH = "E:\\LICENTA\\IRMAS-TrainingData"
OUT_PATH = "../statistics/irmas_complete_features_novoice.csv"
SAMPLE_RATE = 22050
TRACK_DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

FEATURE_NAMES = ["chroma_stft", "rms", "spectral_centroid", "spectral_bandwidth", "rolloff", "zero_crossing_rate", "harmony"]
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

    for folder_name in os.listdir(DATASET_PATH):
        if folder_name != "voi":
            class_path = os.path.join(DATASET_PATH, folder_name)
            for file_name in os.listdir(class_path):
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

    with open(OUT_PATH, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

extract_features_from_data(DATASET_PATH, OUT_PATH)