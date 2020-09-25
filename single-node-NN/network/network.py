from numpy import random, dot, tanh


class Network:
    def __init__(self):
        # Using defined seeds to ensure it will generate same weights at the start
        random.seed(1)
        self.weight_matrix = 2 * random.random((3, 1)) - 1

    # We will use tanh as the activation function
    @staticmethod
    def tanh(x):
        return tanh(x)

    # Derivative of tanh which will be needed in gradient descent
    @staticmethod
    def tanh_derivative(x):
        return 1 - (tanh(x) ** 2)

    # Find the adjustments
    @staticmethod
    def adjustments(self, inputs, weight_matrix):
        return inputs * (1 - (tanh(dot(inputs, weight_matrix)) ** 2)) * (tanh(dot(inputs, weight_matrix)) - 1)

    # Cost Function
    @staticmethod
    def mean_square_error(self, errors):
        mse = 0
        for error in errors:
            mse += error ** 2
        return mse/len(errors)

    # Forward propagation method
    def forward_propagation(self, inputs):
        return self.tanh(dot(inputs, self.weight_matrix))

    # Training the neural network
    def train(self, train_inputs, train_outputs, learning_rate, iterations):
        for iteration in range(iterations):
            outputs = self.forward_propagation(train_inputs)

            # Linear Cost function C = Y(obtained) - Y(desired)
            errors = train_outputs - outputs
            mean_sq_error = self.mean_square_error(errors)

            adjustment = self.adjustments(train_inputs, self.weight_matrix)

            # todo:  Was working on the adjustment values. Go through the Small copy to implement the Adjustment

            # Multiply the error by input and then by gradient of tanh to calculate
            # the adjustment needed for weights
            adjustment = dot(train_inputs.T, error*self.tanh_derivative(output))

            self.weight_matrix += adjustment
