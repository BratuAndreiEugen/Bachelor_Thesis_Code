import os
import pickle

import librosa
import numpy as np
import pandas as pd
from keras.src.utils import to_categorical
from python_speech_features import mfcc
from tqdm import tqdm
from utils.config import InstrumentConfig


class InstrumentDataManipulator:
    def __init__(self, config):
        self.config = config

    def check_data(self):
        """
        Checks if there is a pickle with the randomly generated input data already
        :return: pickle if it exists, otherwise None
        """
        if os.path.isfile(self.config.p_path):
            print("Loading existing data for {} model".format(self.config.mode))
            with open(self.config.p_path, 'rb') as handle:
                tmp = pickle.load(handle)
                return tmp
        else:
            return None

    def build_random_data(self, n_samples, class_dist, prob_dist, df, classes):
        """
        Builds the training input for the neural networks by extracting random config.step/sample_rate of a second samples form all classes based on their probability distribution.
        For each chosen sample it generates the feature using Mel Frequency Cepstral Coeficients (MFCCs) and adds them to the training data.
        Each sample data is reshaped to fit the inputs of the specific neural network used
        :param n_samples: number of samples to be randomly chosen via distribution
        :param class_dist: class distribution
        :param prob_dist: probability distribution
        :param df: dataframe containing label and filename data
        :param classes: list of class names to form the y input of the neural networks
        :return:
        """
        tmp = self.check_data()
        if tmp:
            return tmp.data[0], tmp.data[1]

        X = []
        y = []
        _min, _max = float('inf'), -float('inf')

        for _ in tqdm(range(n_samples)):
            rand_class = np.random.choice(class_dist.index, p=prob_dist)
            file = np.random.choice(df[df.label == rand_class].index)
            wav, rate = librosa.load('clean/' + file, sr=None)
            label = df.at[file, 'label']
            rand_index = np.random.randint(0, wav.shape[
                0] - self.config.step)  # we extract config.step ( usually 1/10th of a sec ) so that we don't reach EOF
            sample = wav[rand_index:rand_index + self.config.step]
            X_sample = mfcc(sample, rate, numcep=self.config.nfeat,
                            nfilt=self.config.nfilt, nfft=self.config.nfft)
            _min = min(np.amin(X_sample), _min)  # amin - min for matrix
            _max = max(np.amax(X_sample), _max)
            X.append(X_sample)
            y.append(classes.index(label))

        self.config.min = _min
        self.config.max = _max

        X, y = np.array(X), np.array(y)
        X = (X - _min) / (_max - _min)

        if self.config.mode == 'conv':
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        elif self.config.mode == 'recurrent':
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
        y = to_categorical(y, num_classes=10)  # categorical cross entropy
        self.config.data = (X, y)

        # Saving the config object in a pickle
        with open(self.config.p_path, 'wb') as handle:
            pickle.dump(self.config, handle, protocol=2)

        return X, y

    def build_inputs_for_file(self, filename):
        wav, rate = librosa.load(filename, sr=self.config.rate)
        x_samples = []  # we categorise for every 1/10 of a second
        index = 0
        for index in range(0, wav.shape[0] - self.config.step, self.config.step):
            current_sample = wav[index:index + self.config.step]
            mfcc_sample = mfcc(current_sample, rate, numcep=self.config.nfeat, nfilt=self.config.nfilt,
                               nfft=self.config.nfft)
            mfcc_sample = np.array(mfcc_sample)
            mfcc_sample = (mfcc_sample - self.config.min) / (self.config.max - self.config.min)
            if self.config.mode == 'conv':
                mfcc_sample = mfcc_sample.reshape(1, mfcc_sample.shape[0], mfcc_sample.shape[1], 1)
            elif self.config.mode == 'recurrent':
                mfcc_sample = np.expand_dims(mfcc_sample, axis=0)
            x_samples.append(mfcc_sample)  # se va chema predict pe fiecare element din lista

        return x_samples

class IRMASDataManipulator:
    def __init__(self, config):
        self.config = config

    def check_data(self):
        """
        Checks if there is a pickle with the randomly generated input data already
        :return: pickle if it exists, otherwise None
        """
        if os.path.isfile(self.config.pickle_path):
            print("Loading existing data for {} model".format(self.config.mode))
            with open(self.config.pickle_path, 'rb') as handle:
                tmp = pickle.load(handle)
                return tmp
        else:
            return None
    def build_data_for_CNN(self, dataset_path, df, classes, input_type='pysf'):
        tmp = self.check_data()
        if tmp:
            return tmp.data[0], tmp.data[1]

        X = []
        y = []
        _min, _max = float('inf'), -float('inf')

        for index, row in tqdm(df.iterrows()):
            file = os.path.join(dataset_path, index)
            label = row["label"]
            wav, rate = librosa.load(file, sr=self.config.rate)
            if input_type == 'pysf':
                X_sample = mfcc(wav, rate, numcep=self.config.nfeat,
                                nfilt=self.config.nfilt, nfft=self.config.nfft)
            else:
                X_sample = librosa.feature.mfcc(y=wav,sr=rate, n_mfcc=self.config.nfeat)
            _min = min(np.amin(X_sample), _min)  # amin - min for matrix
            _max = max(np.amax(X_sample), _max)
            X.append(X_sample)
            y.append(classes.index(label))

        self.config.min = _min
        self.config.max = _max

        X, y = np.array(X), np.array(y)
        X = (X - _min) / (_max - _min)

        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

        y = to_categorical(y, num_classes=len(classes))  # categorical cross entropy
        self.config.data = (X, y)

        # Saving the config object in a pickle
        with open(self.config.pickle_path, 'wb') as handle:
            pickle.dump(self.config, handle, protocol=2)

        return X, y

    def build_random_data_for_CNN(self, dataset_path, df, classes, class_dist, prob_dist, n_samples, input_type='pysf'):
        tmp = self.check_data()
        if tmp:
            return tmp.data[0], tmp.data[1]

        X = []
        y = []
        _min, _max = float('inf'), -float('inf')

        for _ in tqdm(range(n_samples)):
            rand_class = np.random.choice(class_dist.index, p=prob_dist)
            file = np.random.choice(df[df.label == rand_class].index)
            wav, rate = librosa.load(dataset_path + "\\" + file, sr=self.config.rate)
            label = df.at[file, 'label']
            rand_index = np.random.randint(0, wav.shape[
                0] - self.config.step)  # we extract config.step ( usually 1/10th of a sec ) so that we don't reach EOF
            sample = wav[rand_index:rand_index + self.config.step]
            if input_type == 'pysf':
                X_sample = mfcc(sample, rate, numcep=self.config.nfeat,
                                nfilt=self.config.nfilt, nfft=self.config.nfft)
            else:
                X_sample = librosa.feature.mfcc(y=sample,sr=rate, n_mfcc=self.config.nfeat)
            _min = min(np.amin(X_sample), _min)  # amin - min for matrix
            _max = max(np.amax(X_sample), _max)
            X.append(X_sample)
            y.append(classes.index(label))

        self.config.min = _min
        self.config.max = _max

        X, y = np.array(X), np.array(y)
        X = (X - _min) / (_max - _min)

        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

        y = to_categorical(y, num_classes=len(classes))  # categorical cross entropy
        self.config.data = (X, y)

        # Saving the config object in a pickle
        with open(self.config.pickle_path, 'wb') as handle:
            pickle.dump(self.config, handle, protocol=2)

        return X, y

    def build_trucated_data(self, dataset_path, df, classes, input_type='pysf'):
        tmp = self.check_data()
        if tmp:
            return tmp.data[0], tmp.data[1]

        X = []
        y = []
        _min, _max = float('inf'), -float('inf')

        for index, row in tqdm(df.iterrows()):
            file = os.path.join(dataset_path, index)
            label = row["label"]
            wav, rate = librosa.load(file, sr=self.config.rate)
            for i in range(0, wav.shape[0] - self.config.step, self.config.step):
                sample = wav[i:i + self.config.step]
                if input_type == 'pysf':
                    X_sample = mfcc(sample, rate, numcep=self.config.nfeat,
                                    nfilt=self.config.nfilt, nfft=self.config.nfft)
                else:
                    X_sample = librosa.feature.mfcc(y=sample, sr=rate, n_mfcc=self.config.nfeat)
                _min = min(np.amin(X_sample), _min)  # amin - min for matrix
                _max = max(np.amax(X_sample), _max)
                X.append(X_sample)
                y.append(classes.index(label))

        self.config.min = _min
        self.config.max = _max

        X, y = np.array(X), np.array(y)
        X = (X - _min) / (_max - _min)

        if self.config.mode == 'CNN_One':
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        elif self.config.mode == 'RNN_One':
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
        y = to_categorical(y, num_classes=len(classes))  # categorical cross entropy
        self.config.data = (X, y)

        # Saving the config object in a pickle
        with open(self.config.pickle_path, 'wb') as handle:
            pickle.dump(self.config, handle, protocol=2)

        return X, y

    def build_inputs_for_file(self, filename):
        wav, rate = librosa.load(filename, sr=self.config.rate)
        x_samples = []  # we categorise for every 1/10 of a second
        index = 0
        for index in range(0, wav.shape[0] - self.config.step, self.config.step):
            current_sample = wav[index:index + self.config.step]
            mfcc_sample = mfcc(current_sample, rate, numcep=self.config.nfeat, nfilt=self.config.nfilt,
                               nfft=self.config.nfft)
            mfcc_sample = np.array(mfcc_sample)
            mfcc_sample = (mfcc_sample - self.config.min) / (self.config.max - self.config.min)
            if self.config.mode == 'conv':
                mfcc_sample = mfcc_sample.reshape(1, mfcc_sample.shape[0], mfcc_sample.shape[1], 1)
            elif self.config.mode == 'recurrent':
                mfcc_sample = np.expand_dims(mfcc_sample, axis=0)
            x_samples.append(mfcc_sample)  # se va chema predict pe fiecare element din lista

        return x_samples


class DataPreprocessor:
    def __init__(self):
        pass

    @staticmethod
    def envelope(y, rate, threshold, window_size=1 / 10):
        """
        Envelopes the audio in a mask using a rolling window
        :param y: signal values
        :param rate: sample rate of the signal
        :param threshold: threshold for the decision of applying the mask or not
        :param window_size: length of the rolling window
        :return:
        """
        mask = []
        y = pd.Series(y).apply(np.abs)
        y_mean = y.rolling(window=int(rate * window_size), min_periods=1,
                           center=True).mean()  # window_size = 1/10th of a second ( default )
        for mean in y_mean:
            if mean > threshold:
                mask.append(True)
            else:
                mask.append(False)

        return mask
