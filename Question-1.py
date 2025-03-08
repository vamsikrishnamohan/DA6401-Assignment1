import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import fashion_mnist
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.datasets import mnist

# Load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Initialize wandb
wandb.init(project="DL-Assignment-1", name="Class Samples")

# Define a function to plot samples
def plot_samples_for_index(sample_index):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for class_id in range(10):
        # Find indices of samples belonging to this class
        class_indices = np.where(y_train == class_id)[0]

        # Make sure we don't exceed array bounds by using modulo
        safe_index = sample_index % len(class_indices)
        selected_idx = class_indices[safe_index]

        # Display the image
        axes[class_id].imshow(X_train[selected_idx], cmap='gray')
        axes[class_id].set_title(f"{class_names[class_id]}\nSample #{safe_index}")
        axes[class_id].axis('off')

    plt.tight_layout()
    return fig

# Define a range of indices to explore
sample_indices = list(range(0, 35, 5))  # From 0 to 35, steps of 5

# Log multiple visualizations for different sample indices
for idx in sample_indices:
    fig = plot_samples_for_index(idx)

    # Log to wandb with the specific index as the step
    wandb.log({
        "Class Samples": wandb.Image(fig)
    }, step=idx)

    plt.close(fig)

wandb.finish()