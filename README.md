# DA6401 Assignment 1
Assignment 1 submission for the course DA6401 Deep Learning.

S.Vamsi Krishna Mohan (DA24M026)
---

Assignment Report :[WandB report link](https://api.wandb.ai/links/krishvamsi321-indian-institute-of-technology-madras/maixp1aa)

### Install the required dependencies using the requirements.txt file:
```python
   pip install -r requirements.txt
```
## Running the code

### For Question 1: visualization of samples
     python Question-1.py

### For Training:
     python train.py

### For confusion matrix of best model and its accuracy:
     python Question-7.py

### For 3 best models on MNIST dataset :
     python Question-10.py  


## Question 1
The code for question 1 can be accessed [here](https://github.com/vamsikrishnamohan/DA6401-Assignment1/blob/main/Question-1.py). The program, reads the data from `keras.datasets`, picks one example from each class and logs the same to `wandb`.

## Questions 2-4
The neural network is implemented by the class `NeuralNetwork`, present in the `nueral_network.py` and 'back_propagation.py' files.  
### Building a `NeuralNetwork`
An instance of `NeuralNetwork` is as follows:
```Python
 model = NueralNetwork(
            input_size=784,
            hidden_layers=[config.hidden_layer_size] * config.hidden_layers,
            output_size=10,
            activation=config.activation,
            init_method=config.init_method
        ) 
```
It can be implemented by passing the following values:

- **layers**  
    An example of layer is as follows:
    
    ```python
    model = [
     "hidden_layers": [128, 128],
        "activation": "relu",
        "init_method": "xavier",
        "learning_rate": 0.0001,
        "optimizer": "adam",
        "batch_size": 32,
        "epochs": 10,
        "weight_decay": 0
       ]
    ```

- **batch_size**  
    The Batch Size is passed as an integer that determines the size of the mini batch to be taken into consideration.

- **optimizer**  
    The optimizer value is passed as a string, that is internally converted into an instance of the specified optimizer class. The optimizer classes are present inside the file `optimizers.py`. An instance of the class can be created by passing the corresponding parameters:
    + Normal: eta   
        (default: eta=0.01)
    + Momentum: eta, gamma   
        (default: eta=1e-3, gamma=0.9)
    + Nesterov: eta, gamma   
        (default: eta=1e-3, gamma=0.9)
    + AdaGrad: eta, eps   
        (default: eta=1e-2, eps=1e-7)
    + RMSProp: beta, eta , eps    
        (default: beta=0.9, eta = 1e-3, eps = 1e-7)
    + Adam: beta1, beta2, eta, eps   
        (default: beta1=0.9, beta2=0.999, eta=1e-2, eps=1e-8)
    + Nadam: beta1, beta2, eta, eps   
        (default: beta1=0.9, beta2=0.999, eta=1e-3, eps=1e-7)

- **intialization**: A string - `"Random"` or `"Xavier"` can be passed to change the initialization of the weights in the model.

- **epochs**: The number of epochs is passed as an integer to the neural network.


- **loss**: The loss type is passed as a string, that is internally converted into an instance of the specified compute_loss function. The optimizer classes are present inside the file `back_propagation.py`. 

- **X_val**: The validation dataset, used to validate the model.


### Training the `NeuralNetwork`
The model can be trained by loading model with model= NueralNetwork(.,.) followed by `model.backpropogation`. It is done as follows:

```python
# Initialize model with the chosen hyperparameters
model = NueralNetwork(
    input_size=784,
    hidden_layers=[config.hidden_layer_size] * config.hidden_layers,
    output_size=10,
    activation=config.activation,
    init_method=config.init_method
)
model.backpropogation()
```

### Testing the `NeuralNetwork`
The model can be tested by calling the model.predict(X_test) member function, with the testing dataset and the expected `y_test`. The `y_test` values are only used for calculating the test accuracy. It is done in the following manner:

```python
 y_test_pred = model.predicte(X_test)
```

## Question 7
The confusion matrix is logged using the following code:

```python
wandb.log({"conf_mat" : wandb.plot.confusion_matrix(
                        y_test,
                        y_test_pred,
                        class_names=["T-shirt/top","Trouser","Pullover",\
                                     "Dress","Coat","Sandal","Shirt","Sneaker",\
                                     "Bag","Ankle boot"])})
```


## Question 10
Three hyperparameter sets were selected and were run on the MNIST dataset. The configurations choosen are as follows:

- Configuration 1:

 ```python
   [
    'optimizer' : 'adam',
    'hidden_layer_size' :128,
    'init_method':'xavier',
    'activation':'relu',
    'hidden_layers' :[3],
    'weight_decay': 0.0005,
    'learning_rate':0.001,
    'epochs':10,
    'batch_size': 64
   ]
```
- Configuration 2:
```python
  [
    'optimizer' : 'adam',
    'hidden_layer_size' :128,
    'init_method':'xavier',
    'activation':'tanh',
    'hidden_layers' :[5],
    'weight_decay': 0,
    'learning_rate':0.001,
    'epochs':10,
    'batch_size': 64
   ]
```
- Configuration 3:
```python
  [
    'optimizer' : 'adam',
    'hidden_layer_size' :64,
    'init_method':'xavier',
    'activation':'relu',
    'hidden_layers' :[3],
    'weight_decay': 0.0005,
    'learning_rate':0.001,
    'epochs':10,
    'batch_size': 64
  ]
```
---
The codes are organized as follows:

| Question | Location | Function | 
|----------|----------|----------|
| Question 1 | [Question-1](https://github.com/vamsikrishnamohan/DA6401-Assignment1/blob/main/Question-1.py) | Logging Representative Images | 
| Question 2 | [Question-2](https://github.com/vamsikrishnamohan/DA6401-Assignment1/blob/main/nueral_network.py) | Feedforward Architecture |
| Question 3 | [Question-3](https://github.com/vamsikrishnamohan/DA6401-Assignment1/blob/main/back_propagation.py) | Complete Neural Network |
| Question 4 | [Question-4](https://github.com/vamsikrishnamohan/DA6401-Assignment1/blob/main/train.py) | 'train.py' Hyperparameter sweeps using `wandb` |
| Question 7 | [Question-7](https://github.com/vamsikrishnamohan/DA6401-Assignment1/blob/main/Question-7.py) | Confusion Matrix logging for the best Run | 
| Question 10 | [Question-10](https://github.com/vamsikrishnamohan/DA6401-Assignment1/blob/main/Question-10.py) | Top 3 best Hyperparameter configurations for MNIST data | 
| Utility File | [utils.py](https://github.com/vamsikrishnamohan/DA6401-Assignment1/blob/main/utils.py)
