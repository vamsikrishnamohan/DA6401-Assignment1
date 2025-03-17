from utils import *
from nueral_network import NeuralNetwork

class NeuralNetwork_with_Backprop(NeuralNetwork):  # Extending the previous class
    def backpropagation(self, X, y, batch_size=32, learning_rate=0.001,
                        optimizer='sgd', epochs=10, beta1=0.9, beta2=0.999,
                        epsilon=1e-8, weight_decay=0,loss_type="CrossEntropy"):

        m = X.shape[0]  # Number of training examples
        n_batches = int(np.ceil(m / batch_size))
        history = {'loss': [], 'accuracy': []}

        # Initialize=ing optimizer parameters
        if optimizer in ['momentum', 'nag']:
            # Velocity for momentum and NAG
            velocities_w = [np.zeros_like(w) for w in self.weights]
            velocities_b = [np.zeros_like(b) for b in self.biases]

        if optimizer in ['rmsprop', 'adam', 'nadam']:
            # Cache for RMSprop, Adam, and Nadam
            cache_w = [np.zeros_like(w) for w in self.weights]
            cache_b = [np.zeros_like(b) for b in self.biases]

        if optimizer in ['adam', 'nadam']:
            # Momentum for Adam and Nadam
            moment_w = [np.zeros_like(w) for w in self.weights]
            moment_b = [np.zeros_like(b) for b in self.biases]
            t = 0  # Timestep for bias correction

        for epoch in range(epochs):
            # Shuffle data for each epoch
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            correct_preds = 0

            for batch in range(n_batches):
                # Get mini-batch
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, m)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                batch_size_actual = X_batch.shape[0]

                # Forward pass
                y_pred = self.forward(X_batch)
                loss = self.compute_loss(y_pred, y_batch,loss_type) 
                epoch_loss += loss * batch_size_actual

                # Calculate accuracy
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(y_batch, axis=1)
                correct_preds += np.sum(y_pred_classes == y_true_classes)

                # Backpropagation - compute gradients
                dA = y_pred - y_batch  # Derivative of softmax with cross-entropy

                dW = []
                db = []

                # Output layer gradients
                dW_last = np.dot(self.A[-2].T, dA) / batch_size_actual
                db_last = np.sum(dA, axis=0, keepdims=True) / batch_size_actual

                # Adding L2 regularization
                if weight_decay > 0:
                    dW_last += weight_decay * self.weights[-1]

                dW.insert(0, dW_last)
                db.insert(0, db_last)

                # Backpropagate through hidden layers
                dA_prev = dA
                for i in reversed(range(len(self.weights) - 1)):
                    # Compute dh for the current layer
                    dH = np.dot(dA_prev, self.weights[i+1].T) * self.activate(self.H[i], derivative=True)

                    # Compute gradients
                    dW_curr = np.dot(self.A[i].T, dH) / batch_size_actual
                    db_curr = np.sum(dH, axis=0, keepdims=True) / batch_size_actual

                    # Add L2 regularization
                    if weight_decay > 0:
                        dW_curr += weight_decay * self.weights[i]

                    dW.insert(0, dW_curr)
                    db.insert(0, db_curr)

                    # Update dA for next iteration
                    dA_prev = dH

                # Update weights and biases based on optimizer
                if optimizer == 'sgd':
                    # Standard SGD
                    for i in range(len(self.weights)):
                        self.weights[i] -= learning_rate * dW[i]
                        self.biases[i] -= learning_rate * db[i]

                elif optimizer == 'momentum':
                    # SGD with Momentum
                    for i in range(len(self.weights)):
                        velocities_w[i] = beta1 * velocities_w[i] + (1 - beta1) * dW[i]
                        velocities_b[i] = beta1 * velocities_b[i] + (1 - beta1) * db[i]

                        self.weights[i] -= learning_rate * velocities_w[i]
                        self.biases[i] -= learning_rate * velocities_b[i]

                elif optimizer == 'nag':
                    # Nesterov Accelerated Gradient
                    for i in range(len(self.weights)):
                        # Update velocity first
                        velocities_w[i] = beta1 * velocities_w[i] + (1 - beta1) * dW[i]
                        velocities_b[i] = beta1 * velocities_b[i] + (1 - beta1) * db[i]

                        # "Look ahead" - use the updated velocity for the gradient step
                        self.weights[i] -= learning_rate * (beta1 * velocities_w[i] + (1 - beta1) * dW[i])
                        self.biases[i] -= learning_rate * (beta1 * velocities_b[i] + (1 - beta1) * db[i])

                elif optimizer == 'rmsprop':
                    # RMSprop
                    for i in range(len(self.weights)):
                        # Update cache (moving average of squared gradients)
                        cache_w[i] = beta2 * cache_w[i] + (1 - beta2) * (dW[i] ** 2)
                        cache_b[i] = beta2 * cache_b[i] + (1 - beta2) * (db[i] ** 2)

                        # Update weights and biases
                        self.weights[i] -= learning_rate * dW[i] / (np.sqrt(cache_w[i]) + epsilon)
                        self.biases[i] -= learning_rate * db[i] / (np.sqrt(cache_b[i]) + epsilon)

                elif optimizer == 'adam':
                    # Adam optimizer
                    t += 1  # Increment timestep

                    for i in range(len(self.weights)):
                        # Update first moment estimate (momentum)
                        moment_w[i] = beta1 * moment_w[i] + (1 - beta1) * dW[i]
                        moment_b[i] = beta1 * moment_b[i] + (1 - beta1) * db[i]

                        # Update second moment estimate (RMSprop)
                        cache_w[i] = beta2 * cache_w[i] + (1 - beta2) * (dW[i] ** 2)
                        cache_b[i] = beta2 * cache_b[i] + (1 - beta2) * (db[i] ** 2)

                        # Bias correction
                        moment_w_corrected = moment_w[i] / (1 - beta1 ** t)
                        moment_b_corrected = moment_b[i] / (1 - beta1 ** t)
                        cache_w_corrected = cache_w[i] / (1 - beta2 ** t)
                        cache_b_corrected = cache_b[i] / (1 - beta2 ** t)

                        # Update weights and biases
                        self.weights[i] -= learning_rate * moment_w_corrected / (np.sqrt(cache_w_corrected) + epsilon)
                        self.biases[i] -= learning_rate * moment_b_corrected / (np.sqrt(cache_b_corrected) + epsilon)

                elif optimizer == 'nadam':
                    # Nadam (Adam with Nesterov momentum)
                    t += 1  # Increment timestep

                    for i in range(len(self.weights)):
                        # Update first moment estimate
                        moment_w[i] = beta1 * moment_w[i] + (1 - beta1) * dW[i]
                        moment_b[i] = beta1 * moment_b[i] + (1 - beta1) * db[i]

                        # Update second moment estimate
                        cache_w[i] = beta2 * cache_w[i] + (1 - beta2) * (dW[i] ** 2)
                        cache_b[i] = beta2 * cache_b[i] + (1 - beta2) * (db[i] ** 2)

                        # Bias correction
                        moment_w_corrected = moment_w[i] / (1 - beta1 ** t)
                        moment_b_corrected = moment_b[i] / (1 - beta1 ** t)
                        cache_w_corrected = cache_w[i] / (1 - beta2 ** t)
                        cache_b_corrected = cache_b[i] / (1 - beta2 ** t)

                        # Nesterov momentum update
                        moment_w_nesterov = beta1 * moment_w_corrected + (1 - beta1) * dW[i] / (1 - beta1 ** t)
                        moment_b_nesterov = beta1 * moment_b_corrected + (1 - beta1) * db[i] / (1 - beta1 ** t)

                        # Update weights and biases
                        self.weights[i] -= learning_rate * moment_w_nesterov / (np.sqrt(cache_w_corrected) + epsilon)
                        self.biases[i] -= learning_rate * moment_b_nesterov / (np.sqrt(cache_b_corrected) + epsilon)

            # Calculate epoch statistics
            epoch_loss /= m
            epoch_accuracy = correct_preds / m

            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_accuracy)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        return history