from keras import regularizers
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed, BatchNormalization, MaxPool2D
from keras.models import Sequential
from matplotlib import pyplot as plt


class Neural_Network:
    def __init__(self):
        pass

    def plot_history(self, history):
        """Plots accuracy/loss for training/validation set as a function of the epochs

            :param history: Training history of model
            :return:
        """
        plt.figure(1)
        plt.plot(history.history["accuracy"], label="train accuracy")
        plt.plot(history.history["val_accuracy"], label="validation accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(loc="lower right")
        plt.title("Accuracy")
        plt.show()

        plt.figure(2)
        plt.plot(history.history["loss"], label="train error")
        plt.plot(history.history["val_loss"], label="validation error")
        plt.ylabel("Error")
        plt.xlabel("Epoch")
        plt.legend(loc="upper right")
        plt.title("Error")
        plt.show()
    def predict_from_file(self, inputs_to_predict):
        pass

class CNN_One(Neural_Network):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.model = None

    def create_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
                         input_shape=self.__input_shape))
        self.model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same'))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same'))
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dense(units=self.__output_shape, activation='softmax'))
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])

        return self.model

class RNN_One(Neural_Network):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.model = None

    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=128, return_sequences=True,
                       input_shape=self.__input_shape))  # Long Short Term Memory Unit ( like Dense layer )
        self.model.add(LSTM(units=128, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(TimeDistributed(Dense(units=64, activation='relu')))
        self.model.add(TimeDistributed(Dense(units=32, activation='relu')))
        self.model.add(TimeDistributed(Dense(units=16, activation='relu')))
        self.model.add(TimeDistributed(Dense(units=8, activation='relu')))
        self.model.add(Flatten())
        self.model.add(Dense(self.__output_shape, activation='softmax'))
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])

        return self.model

class ANN_One(Neural_Network):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.model = None

    def create_model(self):
        self.model = Sequential()

        self.model.add(Flatten(input_shape=self.__input_shape))
        self.model.add(Dense(units=512, activation='relu'))
        self.model.add(Dense(units=1024, activation='relu'))
        self.model.add(Dense(units=512, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.__output_shape, activation='softmax'))
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])

        return self.model

class LSTM_One(Neural_Network):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.model = None

    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=128, return_sequences=False, input_shape=self.__input_shape))
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=32, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.__output_shape, activation='softmax'))
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return self.model

class CNN_Two(Neural_Network):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.model = None
    def create_model(self):
        self.model = Sequential()

        # Convolutional layers
        self.model.add(Conv2D(16, (3, 3), activation='relu', input_shape=self.__input_shape, padding='same',
                         kernel_regularizer=regularizers.l2(0.01)))
        self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.01)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.01)))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.01)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Dropout(0.25))

        # Flatten layer
        self.model.add(Flatten())

        # Dense layers
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(self.__output_shape, activation='softmax'))

        self.model.summary()

        # Compile the self.model
        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return self.model