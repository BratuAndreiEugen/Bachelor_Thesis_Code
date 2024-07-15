import io

import librosa
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class FeatureExtractor:
    def __init__(self):
        pass

    @staticmethod
    def extract_feature(y, sr, feature_name):
        # Get the feature extraction function from librosa.feature
        feature_function = getattr(librosa.feature, feature_name, None)

        if feature_function is None:
            raise ValueError(f"Feature '{feature_name}' is not available in librosa.feature")

        # Extract the feature using the specified function
        if feature_name == "zero_crossing_rate" or feature_name == "rms":
            feature = feature_function(y=y)
        else:
            feature = feature_function(y=y, sr=sr)

        return feature

    @staticmethod
    def extract_mfcc(y, sr, n_mfcc=13):
        feature = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return feature

    @staticmethod
    def normalize(x, axis=0):
        return minmax_scale(x, axis=axis)

    @staticmethod
    def extract_mel_spec(y, sr):
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        feature = librosa.power_to_db(mel_spec, ref=np.max)
        return feature

    @staticmethod
    def extract_chroma_stft(y, sr):
        feature = librosa.feature.chroma_stft(y=y, sr=sr)
        return feature

    @staticmethod
    def save_1d_feature_plot(y, sr, feature):
        """
        For 1d Features (ex. Spectral Centroid)
        :param y:
        :param sr:
        :param feature:
        :return:
        """
        frames = range(len(feature[0]))
        t = librosa.frames_to_time(frames, sr=sr)

        fig, ax = plt.subplots(figsize=(16, 6))
        canvas = FigureCanvas(fig)

        librosa.display.waveshow(y, sr=sr, alpha=0.6, color='blue', ax=ax, label='Waveform')

        ax.plot(t, minmax_scale(feature[0], axis=0), color='red', linewidth=2, label='Normalized Spectral Centroids')

        # Enhancing the plot
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Save the plot to a PIL image
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        img = Image.open(buf)

        return img

    @staticmethod
    def save_2d_feature_plot(y, sr, feature):
        """
        For MFCC & Chroma STFT
        :param y:
        :param sr:
        :param feature:
        :return:
        """
        fig, ax = plt.subplots(figsize=(16, 6))
        canvas = FigureCanvas(fig)

        librosa.display.specshow(feature, sr=sr, fmax=8000, ax=ax)

        # Save the plot to a PIL image
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        img = Image.open(buf)

        return img

    def make_image_list_for_input(self, audio_file, feature_name, sr=22050, step=3):
        y, sr = librosa.load(audio_file, sr=22050)
        image_list = []
        for index in range(0, len(y) - int(step * sr), int(step * sr)):
            sample = y[index:index + int(step * sr)]
            if feature_name == "13 MFCC":
                feature = self.extract_mfcc(sample, sr, n_mfcc=13)
                image_list.append(self.save_2d_feature_plot(sample, sr, feature))
            elif feature_name == "20 MFCC":
                feature = self.extract_mfcc(sample, sr, n_mfcc=20)
                image_list.append(self.save_2d_feature_plot(sample, sr, feature))
            elif feature_name == "Mel Spectrogram":
                feature = self.extract_mel_spec(sample, sr)
                image_list.append(self.save_2d_feature_plot(sample, sr, feature))
            elif feature_name == "Chroma STFT":
                feature = self.extract_chroma_stft(sample, sr)
                image_list.append(self.save_2d_feature_plot(sample, sr, feature))
            elif feature_name == "Harmony":
                feature = self.extract_feature(sample, sr, feature_name="chroma_cens")
                image_list.append(self.save_1d_feature_plot(sample, sr, feature))
            else:
                feature_func = feature_name.lower().replace(" ", "_")
                feature = self.extract_feature(sample, sr, feature_name=feature_func)
                image_list.append(self.save_1d_feature_plot(sample, sr, feature))

        return image_list
