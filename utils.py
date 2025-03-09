import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.datasets import mnist

import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns


from nueral_network import NueralNetwork
from back_propagation import NueralNetwork

import argparse
