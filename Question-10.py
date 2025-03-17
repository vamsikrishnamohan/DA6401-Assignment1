from utils import * 
from nueral_network import NeuralNetwork
from back_propagation import NeuralNetwork_with_Backprop

###############################################################
# Running the best model on MNIST dataset. 
###############################################################

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data
X_train_full = X_train_full.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Reshape data (flatten 28x28 to 784)
X_train_full = X_train_full.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

# Split into training and validation sets (10% for validation)
X_train, X_val, y_train, y_val = train_test_split(
	X_train_full, y_train_full, test_size=0.1, random_state=42
)

# One-hot encode labels
y_train_one_hot = np.eye(10)[y_train]
y_val_one_hot = np.eye(10)[y_val]

# Three Best configurations from all the configurations that have high accuracy on MNIST dataset:

###############################################################
# Configuration 1: optimizer = Adam, init = Xavier, activation = RELU, hidden_layer_size = 128, batch_size = 64,  num_hidden_layers = 3
###############################################################
config={
    'optimizer' : 'adam',
    'hidden_layer_size' :128,
    'init_method':'xavier',
    'activation':'relu',
    'hidden_layers' :[3],
    'weight_decay': 0.0005,
    'learning_rate':0.001,
    'epochs':10,
    'batch_size': 64
}
model1 = NeuralNetwork_with_Backprop(
            input_size=784,
            hidden_layers=[config.hidden_layer_size] * config.hidden_layers,
            output_size=10,
            activation=config.activation,
            init_method=config.init_method
        )

        # Train model
history = model1.backpropagation(
            X_train, y_train_one_hot,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer,
            epochs=config.epochs,
            weight_decay=config.weight_decay,
            loss_type="CrossEntropy"
        )

###############################################################
y_pred = model1.predict(X_test)
if len(y_pred.shape) == 1:
        # If predictions are already class indices
        y_pred_classes = y_pred
else:
        # If predictions are class probabilities
        y_pred_classes = np.argmax(y_pred, axis=1)
    
# y_pred_classes should be an integer array
y_pred_classes = y_pred_classes.astype(int)

test_accuracy = np.mean(y_pred_classes == y_test)
print(f"Test accuracy of Model-1: {test_accuracy:.2%}")

###############################################################
# ## Configuration 2: optimizer = Adam, init = Xavier, activation = tanh, hidden_layer_size = 128, batch_size = 64, num_hidden_layers = 5
###############################################################
config={
    'optimizer' : 'adam',
    'hidden_layer_size' :128,
    'init_method':'xavier',
    'activation':'tanh',
    'hidden_layers' :[5],
    'weight_decay': 0,
    'learning_rate':0.001,
    'epochs':10,
    'batch_size': 64
}
model2= NeuralNetwork_with_Backprop(
            input_size=784,
            hidden_layers=[config.hidden_layer_size] * config.hidden_layers,
            output_size=10,
            activation=config.activation,
            init_method=config.init_method
        )

# Train model
history = model2.backpropagation(
            X_train, y_train_one_hot,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer,
            epochs=config.epochs,
            weight_decay=config.weight_decay,
            loss_type="CrossEntropy"
        )

###############################################################
y_pred = model2.predict(X_test)
if len(y_pred.shape) == 1:
        # If predictions are already class indices
        y_pred_classes = y_pred
else:
        # If predictions are class probabilities
        y_pred_classes = np.argmax(y_pred, axis=1)
    
# y_pred_classes should be an integer array
y_pred_classes = y_pred_classes.astype(int)

test_accuracy = np.mean(y_pred_classes == y_test)
print(f"Test accuracy of Model-1: {test_accuracy:.2%}")

###############################################################
# Configuration 3: optimizer = Adam, init = Xavier, activation = relu, hidden_layer_size = 64, batch_size = 64, num_hidden_layers = 3
###############################################################
config={
    'optimizer' : 'adam',
    'hidden_layer_size' :64,
    'init_method':'xavier',
    'activation':'relu',
    'hidden_layers' :[3],
    'weight_decay': 0.0005,
    'learning_rate':0.001,
    'epochs':10,
    'batch_size': 64
}
model3 = NeuralNetwork_with_Backprop(
            input_size=784,
            hidden_layers=[config.hidden_layer_size] * config.hidden_layers,
            output_size=10,
            activation=config.activation,
            init_method=config.init_method
        )

# Train model
history = model3.backpropagation(
            X_train, y_train_one_hot,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer,
            epochs=config.epochs,
            weight_decay=config.weight_decay,
            loss_type="CrossEntropy"
        )

###############################################################
y_pred = model3.predict(X_test)
if len(y_pred.shape) == 1:
        # If predictions are already class indices
        y_pred_classes = y_pred
else:
        # If predictions are class probabilities
        y_pred_classes = np.argmax(y_pred, axis=1)
    
# y_pred_classes should be an integer array
y_pred_classes = y_pred_classes.astype(int)

test_accuracy = np.mean(y_pred_classes == y_test)
print(f"Test accuracy of Model-1: {test_accuracy:.2%}")
