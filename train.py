from utils import *
from nueral_network import NueralNetwork

def generate_sweep_name(config):
    return f"hl_{config.hidden_layers}_bs_{config.batch_size}_ac_{config.activation}_MSE_"

def run_wandb_sweep():
    # sweep configuration
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'hidden_layers': {
                'values': [3, 4, 5]
            },

            'hidden_layer_size': {
                'values': [32, 64, 128]
            },

            'activation': {
                'values': ['sigmoid', 'tanh', 'relu']
            },
            'optimizer': {
                'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']
            },
            'learning_rate': {
                'distribution': 'log_uniform',
                'min': np.log(1e-4),
                'max': np.log(1e-3)
            },
            'batch_size': {
                'values': [16, 32, 64]
            },
            'epochs': {
                'values': [5, 10]
            },
            'weight_decay': {
                'values': [0, 0.0005, 0.5]
            },
            'init_method': {
                'values': ['random', 'xavier']
            }
        } 
        
    }

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="DL-Assignment-1")

    def train_model():
        # Initialize wandb run
        run = wandb.init(project="DL-Assignment-1")
        config = wandb.config
        run.name = generate_sweep_name(config)

        # Load data
        (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

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

        # Initialize model with the chosen hyperparameters
        model = NueralNetwork(
            input_size=784,
            hidden_layers=[config.hidden_layer_size] * config.hidden_layers,
            output_size=10,
            activation=config.activation,
            init_method=config.init_method
        )

        # Train model
        history = model.backpropagation(
            X_train, y_train_one_hot,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer,
            epochs=config.epochs,
            weight_decay=config.weight_decay,
            loss_type="CrossEntropy"
        )

        # Evaluate on validation set
        val_predictions = model.predict(X_val)
        val_accuracy = np.mean(val_predictions == y_val)
        best_val_accuracy=0

        # Log metrics
        for epoch in range(config.epochs):
            val_loss = model.compute_loss(model.forward(X_val), np.eye(10)[y_val],loss_type="CrossEntropy")

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": history['loss'][epoch],
                "train_accuracy": history['accuracy'][epoch],
                "val_loss": val_loss,
                
            })

        wandb.log({"val_accuracy": val_accuracy})

        run.finish()

    # Run the sweep
    wandb.agent(sweep_id, train_model,count=50)  # count=50 runs 50 trails of different sweep configurations 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training a deep Neural Network with Weights & Biases sweep.")
    
    parser.add_argument('-wp', '--wandb_project', type=str, default='DL-Assignment-1', help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', type=str, default='krishvamsi321', help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
    parser.add_argument('-d', '--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='fashion_mnist', help='Dataset to use for training')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train neural network')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size used to train neural network')
    parser.add_argument('-l', '--loss', type=str, choices=['mean_squared_error', 'cross_entropy'], default='cross_entropy', help='Loss function to use')
    parser.add_argument('-o', '--optimizer', type=str, choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], default='adam', help='Optimizer to use')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--momentum', type=float, default=0.5, help='Momentum used by momentum and nag optimizers')
    parser.add_argument('-beta', '--beta', type=float, default=0.5, help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5, help='Beta1 used by adam and nadam optimizers')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5, help='Beta2 used by adam and nadam optimizers')
    parser.add_argument('-eps', '--epsilon', type=float, default=0.000001, help='Epsilon used by optimizers')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0, help='Weight decay used by optimizers')
    parser.add_argument('-w_i', '--weight_init', type=str, choices=['random', 'xavier'], default='xavier', help='Weight initialization method')
    parser.add_argument('-nhl', '--num_layers', type=int, default=5, help='Number of hidden layers used in feedforward neural network')
    parser.add_argument('-sz', '--hidden_size', type=int, default=128, help='Number of hidden neurons in a feedforward layer')
    parser.add_argument('-a', '--activation', type=str, choices=['identity', 'sigmoid', 'tanh', 'ReLU'], default='relu', help='Activation function to use')
    
    args = parser.parse_args()
    run_wandb_sweep(args)