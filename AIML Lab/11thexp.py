import numpy as np

class NeuralNetwork:
    def __init__(self):
        # Seeding for random number generation
        np.random.seed(1)

        # Converting weights to a 3x1 matrix with values from -1 to 1
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        # Applying the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Computing derivative of the sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        # Training the model to make accurate predictions
        for iteration in range(training_iterations):

            # Forward pass
            output = self.think(training_inputs)

            # Error calculation
            error = training_outputs - output

            # Weight adjustment
            adjustments = np.dot(
                training_inputs.T,
                error * self.sigmoid_derivative(output)
            )

            self.synaptic_weights += adjustments

    def think(self, inputs):
        # Passing inputs through the neural network
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


if __name__ == "__main__":
    # Initializing the neural network
    neural_network = NeuralNetwork()

    print("Beginning Randomly Generated Weights:")
    print(neural_network.synaptic_weights)

    # Training data
    training_inputs = np.array([
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1]
    ])

    training_outputs = np.array([[0, 1, 1, 0]]).T

    # Training the network
    neural_network.train(training_inputs, training_outputs, 15000)

    print("Ending Weights After Training:")
    print(neural_network.synaptic_weights)

    # User input
    user_input_one = float(input("User Input One: "))
    user_input_two = float(input("User Input Two: "))
    user_input_three = float(input("User Input Three: "))

    print("Considering New Situation:",
          user_input_one, user_input_two, user_input_three)

    print("New Output data:")
    print(neural_network.think(
        np.array([user_input_one, user_input_two, user_input_three])
    ))

    print("Wow, we did it!")
