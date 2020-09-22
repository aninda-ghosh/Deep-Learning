import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# We will be using GPU to train our model. I have a Nvidia GTX 1650 card with compute capability of 7.5 and 896 CUDA
# cores. Hence let's find how the device is referenced.
if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
    print('Please install GPU version of TF')

# We will load the breast cancer data from the sklearn datasets Line 2
data = load_breast_cancer()

# Let's see how the data type looks like and let's see what features are there.
print(type(data))
print(data.keys())
print(data.data.shape)
print(data.target)
print(data.target_names)
print(data.target.shape)
print(data.feature_names)

# Since we also want to evaluate the model hence we will split the dataset into Training and Validation datasets
X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, test_size=0.33)
N, D = X_train.shape
print(N, D)

# We need to normalize the data so that 1 big value feature does not overshadow the other features.
# todo: But why is Normalisation needed????
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Now it's time to define our model
# We need 1 Input layer and 1 Dense layer and for the Dense layer we need an activation function
# Since we need to bring some non linearity in the activation sigmoid seems perfect match.
# todo: But why bring non-linearity????

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Now we need to setup what parameters we want the model to train with.
# todo: What is ADAM? What is Binary cross entropy?
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Now it's time to solve the geometry problem. Let's play Hide and Seek
# Let's find that non linear curve which will somewhat fit with the existing training data
r = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1000)


print("Train Score:", model.evaluate(X_train, Y_train))
print("Train Score:", model.evaluate(X_test, Y_test))

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

# Make predictions
P = model.predict(X_test)
print(P) # they are outputs of the sigmoid, interpreted as probabilities p(y = 1 | x)

# Round to get the actual predictions
# Note: has to be flattened since the targets are size (N,) while the predictions are size (N,1)

P = np.round(P).flatten()
print(P)

# Calculate the accuracy, compare it to evaluate() output
print("Manually calculated accuracy:", np.mean(P == Y_test))
print("Evaluate output:", model.evaluate(X_test, Y_test))