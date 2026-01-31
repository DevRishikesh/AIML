# Import python libraries required in this example
import numpy as np
from scipy.special import expit as activation_function
from scipy.stats import truncnorm

# DEFINE THE NETWORK

# Generate random numbers within a truncated (bounded) normal distribution
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd,
        (upp - mean) / sd,
        loc=mean,
        scale=sd
    )

# Create the ‘Nnetwork’ class
class Nnetwork:
    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate):

        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate

        self.create_weight_matrices()

    def create_weight_matrices(self):
        """Initialize the weight matrices of the neural network"""

        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs(
            (self.no_of_hidden_nodes, self.no_of_in_nodes)
        )

        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs(
            (self.no_of_out_nodes, self.no_of_hidden_nodes)
        )

    def train(self, input_vector, target_vector):
        pass  # Training logic can be added later

    def run(self, input_vector):
        """
        Run the network with an input vector
        """

        # Convert input to column vector
        input_vector = np.array(input_vector, ndmin=2).T

        # Forward propagation
        input_hidden = activation_function(
            self.weights_in_hidden @ input_vector
        )
        output_vector = activation_function(
            self.weights_hidden_out @ input_hidden
        )

        return output_vector


# RUN THE NETWORK AND GET A RESULT

# Initialize network
simple_network = Nnetwork(
    no_of_in_nodes=2,
    no_of_out_nodes=2,
    no_of_hidden_nodes=4,
    learning_rate=0.6
)

# Run network
result = simple_network.run([3, 4])
print(result)
