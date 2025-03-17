from utils import *
from nueral_network import NeuralNetwork
from back_propagation import NeuralNetwork_with_Backprop

#################################################################### 
# #  Getting the Best Model from sweeps and logging the confusion matrix on Test Data 
####################################################################
def test_best_model():

    # Initialize wandb run
    wandb.init(project="DL-Assignment-1", name="best-model-evaluation")
    
    # Load the data
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    # Normalize and reshape the data
    X_test = X_test.astype("float32") / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1)  # Shape to (n_samples, 784)
    X_train = X_train.astype("float32") / 255.0
    X_train = X_train.reshape(X_train.shape[0], -1)
    
    # Define class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Best model from sweeps
    best_model = NeuralNetwork_with_Backprop(
        input_size=784,
        hidden_layers=[128, 128, 128, 128, 128],  # 5 hidden layers of size 128
        output_size=10,
        activation='relu',
        init_method='xavier'
    )
    
    # Log model architecture to wandb
    wandb.config.update({
        "hidden_layers": [128, 128, 128, 128, 128],
        "activation": "relu",
        "init_method": "xavier",
        "learning_rate": 0.00067324,
        "optimizer": "adam",
        "batch_size": 32,
        "epochs": 10,
        "weight_decay": 0
    })
    
    # Train using backpropagation
    best_model.backpropagation(
        X_train, np.eye(10)[y_train],
        batch_size=32,
        learning_rate=0.00067324,
        optimizer='adam',
        epochs=10,
        weight_decay=0
    )
    
    # Get predictions
    y_pred = best_model.predict(X_test)
    
    # Debug information
    print(f"Shape of y_pred: {y_pred.shape}")
    print(f"Type of y_pred: {type(y_pred)}")
    
    # Handle different prediction formats
    if len(y_pred.shape) == 1:
        # If predictions are already class indices
        y_pred_classes = y_pred
    else:
        # If predictions are class probabilities
        y_pred_classes = np.argmax(y_pred, axis=1)
    
    # y_pred_classes should be an integer array
    y_pred_classes = y_pred_classes.astype(int)
    
    # Calculate accuracy
    test_accuracy = np.mean(y_pred_classes == y_test)
    print(f"Test accuracy: {test_accuracy:.2%}")
    
    # Log test accuracy to wandb
    wandb.log({"test_accuracy": test_accuracy})
    
    # Ensure predictions and ground truth have same shape
    print(f"y_pred_classes shape: {y_pred_classes.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # === Confusion Matrix Visualization ===
    cm = confusion_matrix(y_test, y_pred_classes)
    print(f"Confusion Matrix Shape: {cm.shape}")
    print(f"Class Names Length: {len(class_names)}")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Test Accuracy: {test_accuracy:.2%})')
    plt.tight_layout()
    
    # Save confusion matrix locally
    plt.savefig('confusion_matrix.png')
    
    # Log confusion matrix to wandb
    wandb.log({"confusion_matrix": wandb.Image('confusion_matrix.png')})
    
    # Show plot
    plt.show()
    
    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    test_best_model()