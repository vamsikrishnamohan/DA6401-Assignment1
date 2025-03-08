from utils import *

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_layers=[32, 32], output_size=10, activation='relu', init_method='xavier'):

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation
        self.init_method = init_method

        # Architecture with input, hidden, and output layers
        layer_sizes = [input_size] + hidden_layers + [output_size]

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            if init_method == 'xavier':
                # Xavier/Glorot initialization
                scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
                self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale)
            else:
                # Simple random initialization
                self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)

            self.biases.append(np.zeros((1, layer_sizes[i+1])))

    def activate(self, h, derivative=False):
        """Apply activation function"""
        if self.activation == 'sigmoid':
            if derivative:
                return self.sigmoid(h) * (1 - self.sigmoid(h))
            return self.sigmoid(h)

        elif self.activation == 'tanh':
            if derivative:
                return 1 - np.power(self.tanh(h), 2)
            return self.tanh(h)

        elif self.activation == 'relu':
            if derivative:
                return np.where(h > 0, 1, 0)
            return np.maximum(0, h)

    def sigmoid(self, h):
        return 1 / (1 + np.exp(-np.clip(h, -500, 500)))  # Clip to avoid overflow

    def tanh(self, h):
        return np.tanh(h)

    def softmax(self, h):
        # Subtract max for numerical stability
        exp_h = np.exp(h - np.max(h, axis=1, keepdims=True))
        return exp_h / np.sum(exp_h, axis=1, keepdims=True)

    def forward(self, X):
        # Forward propagation

        # Store activations and pre-activations for backpropagation
        self.H = []  # Pre-activations
        self.A = [X]  # Activations (A[0] is the input)

        # Forward through hidden layers
        for i in range(len(self.weights) - 1):
            H = np.dot(self.A[i], self.weights[i]) + self.biases[i]
            self.H.append(H)
            self.A.append(self.activate(H))

        # Output layer with softmax
        H_out = np.dot(self.A[-1], self.weights[-1]) + self.biases[-1]
        self.H.append(H_out)
        output = self.softmax(H_out)
        self.A.append(output)

        return output

    def compute_loss(self, y_pred, y_true,loss_type):
        
        if loss_type=="CrossEntropy":
            m = y_true.shape[0]  # Batch size
            # Add small epsilon to avoid log(0)
            epsilon = 1e-15
            cross_entropy = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
            return cross_entropy
        
        else:
            # Mean squared error loss
            m = y_true.shape[0]
            return np.mean(np.sum((y_pred - y_true) ** 2, axis=1)) / 2

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
