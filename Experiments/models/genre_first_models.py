from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed, BatchNormalization, MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam

from models.instrument_first_models import Neural_Network


class CNNG1(Neural_Network):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.model = None

    def create_model(self):
        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.__input_shape))
        self.model.add(MaxPool2D((3, 3), strides=(2, 2), padding='same'))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPool2D((3, 3), strides=(2, 2), padding='same'))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(32, (2, 2), activation='relu'))
        self.model.add(MaxPool2D((2, 2), strides=(2, 2), padding='same'))
        self.model.add(BatchNormalization())

        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(self.__output_shape, activation='softmax'))

        self.model.summary()
        optimiser = Adam(learning_rate=0.0001)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])

        return self.model


class RNNG1(Neural_Network):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.model = None

    def create_model(self):
        self.model = Sequential()

        self.model.add(LSTM(64, input_shape=self.__input_shape, return_sequences=True))
        self.model.add(LSTM(64))

        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(self.__output_shape, activation='softmax'))

        self.model.summary()
        optimiser = Adam(learning_rate=0.0001)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])

        return self.model