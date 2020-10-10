import tensorflow as tf
import os.path
from os import path
import numpy as np
import matplotlib.pyplot as plt


class DigitClassifier:
    def __init__(self, filename):
        modelexists = os.path.isfile(filename)
        if modelexists:
            # If model present load it,
            self.model = tf.keras.models.load_model(filename)
            print('Using saved model')
            self.training_needed = 0
        else:
            # else
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            print('Generating new model')
            self.training_needed = 1
        self.training_history = []

    def is_training_needed(self):
        return self.training_needed

    def train(self, x_train, y_train, x_test, y_test, no_of_epoch):
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.training_history = self.model.fit(
            x_train,
            y_train,
            validation_data=(
                x_test,
                y_test
            ),
            epochs=no_of_epoch
        )
        return self.training_history

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def summary(self):
        return self.model.summary()

    def save_model(self, filename):
        self.model.save(filename)


if __name__ == '__main__':
    mnist_dataset = tf.keras.datasets.mnist

    # Split the data set into train data and test data
    (x_train_data, y_train_data), (x_test_data, y_test_data) = mnist_dataset.load_data()

    # Normalize the data set
    x_train_data, x_test_data = x_train_data/255.0, x_test_data/255.0

    print(x_train_data.shape)
    print(y_train_data.shape)
    print(x_test_data.shape)
    print(y_test_data.shape)

    # plot first few images
    for i in range(9):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(x_train_data[i], cmap=plt.get_cmap('gray'))
    # show the figure
    plt.show()

    # Initialize the model
    dc = DigitClassifier('saved_model.h5')
    print(dc.summary())

    # If the model is already present. Then no need of any training
    if dc.is_training_needed():
        # Feed the model and train it
        _train_history_ = dc.train(x_train_data, y_train_data, x_test_data, y_test_data, 10)
        dc.save_model('saved_model.h5')

        plt.plot(_train_history_.history['loss'], label='loss')
        plt.plot(_train_history_.history['val_loss'], label='val_loss')
        plt.legend()
        plt.show()

    # Get the evaluation
    print(dc.evaluate(x_test_data, y_test_data))
    



