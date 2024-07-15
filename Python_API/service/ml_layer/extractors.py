import cv2
import librosa as librosa
import numpy as np
import python_speech_features as python_speech_features
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


class FeatureExtractor:
    def __init__(self):
        pass

    @staticmethod
    def add_silence(audio, target_length=3*22050):
        duration = len(audio)
        if duration < target_length:
            silence_duration = target_length - duration
            silence = np.zeros(silence_duration, dtype=audio.dtype)
            audio = np.concatenate((audio, silence))
        return audio

    @staticmethod
    def extract_best_features(file_path, scaler, sampling_rate=22050, step = 3):
        audio, sr = librosa.load(file_path, sr=sampling_rate)
        enhanced_length = int(len(audio) / sampling_rate) + 1
        audio = FeatureExtractor.add_silence(audio, target_length=enhanced_length*22050)
        X_file = []
        for index in range(0, len(audio) - step * sr, step * sr):
            sample = audio[index:index + step * sr]

            chroma_stft = librosa.feature.chroma_stft(y=sample, sr=sr)
            rms = librosa.feature.rms(y=sample)
            spectral_centroids = librosa.feature.spectral_centroid(y=sample, sr=sr)
            spectral_bandwidths = librosa.feature.spectral_bandwidth(y=sample, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=sample, sr=sr)
            zero_crossing_rates = librosa.feature.zero_crossing_rate(y=sample)
            chroma_cens = librosa.feature.chroma_cens(y=sample, sr=sr)
            tempo, _ = librosa.beat.beat_track(y=sample, sr=sr)
            mfccs = librosa.feature.mfcc(y=sample, sr=sr)

            sample_features = []
            sample_features.append(np.mean(chroma_stft))
            sample_features.append(np.var(chroma_stft))
            sample_features.append(np.mean(rms))
            sample_features.append(np.var(rms))
            sample_features.append(np.mean(spectral_centroids))
            sample_features.append(np.var(spectral_centroids))
            sample_features.append(np.mean(spectral_bandwidths))
            sample_features.append(np.var(spectral_bandwidths))
            sample_features.append(np.mean(spectral_rolloff))
            sample_features.append(np.var(spectral_rolloff))
            sample_features.append(np.mean(zero_crossing_rates))
            sample_features.append(np.var(zero_crossing_rates))
            sample_features.append(np.mean(chroma_cens))
            sample_features.append(np.var(chroma_cens))
            sample_features.append(tempo[0])
            for i, mfcc_column in zip(range(20), mfccs):
                sample_features.append(np.mean(mfcc_column))
                sample_features.append(np.var(mfcc_column))

            X_file.append(sample_features)
        print(X_file)
        X_file = np.array(X_file)
        X_file = scaler.transform(X_file)
        return X_file

    @staticmethod
    def extract_mel_specs(file_path, sampling_rate=22050, step = 3):
        audio, sr = librosa.load(file_path, sr=sampling_rate)
        X_file = []
        for index in range(0, len(audio) - step * sr, step * sr):
            sample = audio[index:index + step * sr]
            mel_spec = librosa.feature.melspectrogram(y=sample, sr=sampling_rate)

            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            fig, ax = plt.subplots(figsize=(1.28, 1.3), dpi=100)  # Adjust figure size to 128x130 pixels

            librosa.display.specshow(mel_spec_db, sr=sampling_rate, x_axis='time', y_axis='mel', fmax=8000, ax=ax)
            plt.axis('off')  # Turn off the axis

            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

            output_file = f'temp_mel/temp_mel_spec.jpg'
            plt.savefig(output_file, format='jpg', bbox_inches='tight', pad_inches=0)
            plt.close()

            image_rgb_matrix = cv2.imread(output_file)
            image_rgb_matrix = image_rgb_matrix.astype('float32') / 255.0
            X_file.append(image_rgb_matrix)

        X_file = np.array(X_file)
        print(X_file.shape)
        return X_file

    @staticmethod
    def extract_mfcc_librosa(file_path, sampling_rate=22050, step = 3, num_mfcc=13, n_fft=2048, hop_length=512, cnn=True):
        audio, sr = librosa.load(file_path, sr=sampling_rate)
        X_file = []
        for index in range(0, len(audio) - step * sr, step * sr):
            sample = audio[index:index + step * sr]
            mfcc = librosa.feature.mfcc(y=sample, sr=sampling_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                        hop_length=hop_length)
            mfcc = mfcc.T

            X_file.append(mfcc)

        X_file = np.array(X_file)
        if cnn:
            X_file = X_file.reshape(X_file.shape[0], X_file.shape[1], X_file.shape[2], 1)

        print(X_file.shape)
        return X_file

    @staticmethod
    def extract_mfcc_pysf(file_path, scaler, sampling_rate=22050, step=0.1, cnn=True):
        audio, sr = librosa.load(file_path, sr=sampling_rate)
        X_file = []
        for index in range(0, len(audio) - int(step * sr), int(step * sr)):
            sample = audio[index:index + int(step * sr)]
            mfcc = python_speech_features.mfcc(sample, sampling_rate, numcep=26, nfilt=13, nfft=551)
            X_file.append(mfcc)

        X_file = np.array(X_file)
        X_file = (X_file - scaler["min"])/(scaler["max"] - scaler["min"])
        if cnn:
            X_file = X_file.reshape(X_file.shape[0], X_file.shape[1], X_file.shape[2], 1)

        print(X_file.shape)
        return X_file
