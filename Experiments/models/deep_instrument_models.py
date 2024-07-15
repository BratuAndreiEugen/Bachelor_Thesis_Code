from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout


class VGG_16:
    def __init__(self, input_shape, output_shape):
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.model = None

    def create_model(self):
        """
        input_shape should be (X.shape[1], X.shape[2], 1)
        :return:
        """
        self.model = Sequential([
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=self.__input_shape),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPool2D(pool_size=(1, 2), strides=2),

            Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPool2D(pool_size=(1, 2), strides=2),

            Conv2D(filters=256, kernel_size=(1, 1), activation='relu', padding='same'),
            Conv2D(filters=256, kernel_size=(1, 1), activation='relu', padding='same'),
            Conv2D(filters=256, kernel_size=(1, 1), activation='relu', padding='same'),
            MaxPool2D(pool_size=(1, 2), strides=2),

            Conv2D(filters=512, kernel_size=(1, 1), activation='relu', padding='same'),
            Conv2D(filters=512, kernel_size=(1, 1), activation='relu', padding='same'),
            Conv2D(filters=512, kernel_size=(1, 1), activation='relu', padding='same'),
            MaxPool2D(pool_size=(1, 2), strides=2),

            Conv2D(filters=512, kernel_size=(1, 1), activation='relu', padding='same'),
            Conv2D(filters=512, kernel_size=(1, 1), activation='relu', padding='same'),
            Conv2D(filters=512, kernel_size=(1, 1), activation='relu', padding='same'),
            MaxPool2D(pool_size=(1, 2), strides=2),

            Flatten(),
            Dense(units=4096, activation='relu'),
            Dense(units=4096, activation='relu'),
            Dense(self.__output_shape, activation='softmax')
        ])

        self.model.summary()

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse'])
        return self.model

    def predict_from_file(self, inputs_to_predict):
        pass

class Yolo_Like:
    def __init__(self, input_shape, output_shape):
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.model = None

    def create_model(self):
        self.model = Sequential()

        self.model.add(Conv2D(filters=64, kernel_size=(7,7), activation='relu', padding='same', input_shape=self.__input_shape))
        self.model.add(MaxPool2D(pool_size=(2,2)))
        self.model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
        self.model.add(MaxPool2D(pool_size=(2,2)))
        self.model.add(Conv2D(filters=128, kernel_size=(1,1), activation='relu', padding='same'))
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'))
        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), activation='relu', padding='same'))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'))
        self.model.add(MaxPool2D(pool_size=(1, 2)))
        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), activation='relu', padding='same'))
        self.model.add(Conv2D(filters=512, kernel_size=(1, 1), activation='relu', padding='same'))
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(MaxPool2D(pool_size=(1,2)))
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(Flatten())
        self.model.add(Dense(units=1024, activation='relu'))
        self.model.add(Dropout(0.8))

        self.model.add(Dense(self.__output_shape, activation='softmax'))

        self.model.summary()

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse'])
        return self.model

