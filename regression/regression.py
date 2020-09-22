import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Prepare the data from the downloaded CSV
# Convert it to array of size Nx1
dataset = pd.read_csv("datasets/moore.csv", header=None).values

x = dataset[:, 0].reshape(-1, 1)
y = dataset[:, 1]

# Transform it to a linear logarithmic dataset with
x = x - x.mean()  # This is the deviation
y = np.log(y)  # Get the log of the y value

# Let's create the the model sequentially now
model = keras.Sequential()
model.add(layers.Input(shape=(1,)))
model.add(layers.Dense(1))

# Optimizer used is SGD (Stochastic Gradient Descent) and the cost function is Mean Squared Error
model.compile(optimizer=keras.optimizers.SGD(0.001, 0.9), loss='mse')


# Learning rate scheduler
def schedule(epoch, lr):
    if epoch >= 50:
        return 0.0001
    return 0.001


scheduler = keras.callbacks.LearningRateScheduler(schedule)

r = model.fit(x, y, epochs=200, callbacks=[scheduler])

# yhat = m*x + a0
m = model.layers[0].get_weights()[0][0, 0]

time_for_doubling = np.log(2)/m

print(time_for_doubling)    # This is almost close to 2 which verifies the moor's law

plt.plot(r.history['loss'], label='loss')
plt.show()
