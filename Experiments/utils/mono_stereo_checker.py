import librosa
import numpy as np
from scipy.io import wavfile


def check_type(path):
    sr, y = wavfile.read(path)
    print(sr)
    print(y.shape)


check_type("C:\Proiecte SSD\Licenta\Basic-Fourier-Usage\wavfiles\instruments\\4c4d4797.wav")
check_type("E:\LICENTA\IRMAS-TrainingData-NoVoice\cel\\[cel][cla]0008__1.wav")
check_type("E:\LICENTA\Philarmonia_equalised\cor_anglais\\english-horn_A3_025_forte_normal.mp3")
check_type("E:\LICENTA\marsyas_gtzan_genres\genres_original\classical\\classical_00013.wav")

# Conclusions
# IRMAS - stereo
# Instruments - mono
# Philarmonia - mono
# Genres(GTZAN) - mono

audio, sr = librosa.load("E:\LICENTA\IRMAS-TrainingData-NoVoice\cel\\[cel][cla]0008__1.wav")
mfccs = librosa.feature.mfcc(y=audio, sr=sr)
print(mfccs.shape)
print(mfccs.T.shape)

for i, mfcc_column in zip(range(30), mfccs.T):
    print(np.mean(mfcc_column))
    print(np.var(mfcc_column))
    print(i)