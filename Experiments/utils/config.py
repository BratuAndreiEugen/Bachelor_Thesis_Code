import os


class InstrumentConfig:
    # nfft - window size
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.rate = rate
        self.nfft = nfft
        self.step = int(rate/10) # a tenth of a second
        self.model_path = os.path.join('../instrument_models', mode + '.keras')
        self.p_path = os.path.join('../instrument_pickles', mode + '.p')
        self.min = float('inf')
        self.max = -float('inf')

class IRMASConfig:
    def __init__(self, model_path, pickle_path):
        self.model_path = model_path
        self.pickle_path = pickle_path

class IRMAS_MFCC_Config(IRMASConfig):
    def __init__(self, model_path, pickle_path, mode='CNN_One', nfilt=26, nfeat=13, nfft=1103, rate=22050, step = 1/10):
        super().__init__(model_path, pickle_path)
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.rate = rate
        self.nfft = nfft
        self.mode = mode
        self.step = int(rate * step)

class IRMAS_Simple_Config(IRMASConfig):
    def __init__(self, model_path, pickle_path, mode='SVM', rate=22050, step = 3):
        super().__init__(model_path, pickle_path)
        self.rate = rate
        self.mode = mode
        self.step = int(rate * step)