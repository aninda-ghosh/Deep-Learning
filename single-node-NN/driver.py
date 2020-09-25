from numpy import array
from network.network import Network as NeuralNetwork

if __name__ == "__main__":
    nn = NeuralNetwork()

    print('Random weights at the start of training')
    print(nn.weight_matrix)

    train_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    train_outputs = array([[0, 1, 1, 0]]).T

    nn.train(train_inputs, train_outputs, 200)

    print('New weights after training')
    print(nn.weight_matrix)

    # Test the neural network with a new situation.
    print("Testing network on new examples ->")
    print(nn.forward_propagation(array([1, 1, 1])))
