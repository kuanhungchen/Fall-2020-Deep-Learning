import os
import json
import numpy as np


def ReLU(x):
    """ReLU activation function
    Args:
        x.shape = (1, layer number)
    Return:
        y.shape = (1, layer number)
    """
    y = np.maximum(0, x)
    return y

def ReLU_back(dA, Z):
    """ReLU derivative function
    Args:
        dA.shape = (layer number, )
        Z.shape = (layer number, )
    Return:
        dZ.shap = (layer number, )
    """
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def Softmax(x):
    """Softmax function
    Args:
        x.shape = (batch size, 10)
    Return:
        y.shape = (batch size, 10)
    """
    exps = np.exp(x)
    y = exps / (1e-8 + np.sum(exps))
    return y

def one_hot_encoding(labels):
    """convert input labels to one-hot encoding
    Args:
        labels.shape = (batch size, 1)
    Return
        output.shape = (batch size, 10)
    """
    N = labels.shape[0]
    output = np.zeros((N, 10), dtype=np.int32)
    for i in range(N):
        output[i][int(labels[i])] = 1
    return output


class Model:
    """Classification model for MNIST dataset
    
    Structure:
    
    #   |     Layer name    |   Input size  | Output size
    -------------------------------------------------------
    1   |   Fully connected |   28 * 28     |       h1
    2   |       ReLU        |       h1      |       h1
    3   |   Fully connected |       h1      |       h2
    4   |       ReLU        |       h2      |       h2
    5   |   Fully connected |       h2      |       10
    6   |       Softmax     |       10      |       10
    
    Default setting:
    1) h1 = 128, h2 = 64
    2) Learning rate = 0.005
    3) beta for momentum = 0.9
    """
    def __init__(self, h1=128, h2=64):
        # initial model weights
        self.weights = {
                'W1': np.random.randn(28 * 28, h1) * np.sqrt(1. / (28 * 28)), "b1": np.random.randn(1, h1) * np.sqrt(1. / (28 * 28)),
                'W2': np.random.randn(h1, h2) * np.sqrt(1. / h1), "b2": np.random.randn(1, h2) * np.sqrt(1. / h1),
                'W3': np.random.randn(h2, 10) * np.sqrt(1. / h2), "b3": np.random.randn(1, 10) * np.sqrt(1. / h2)
                }
        # array to store layer outputs
        self.cache = {
                "Z1": np.zeros((1, h1), dtype=float),
                "A1": np.zeros((1, h1), dtype=float),
                "Z2": np.zeros((1, h2), dtype=float),
                "A2": np.zeros((1, h2), dtype=float),
                "Z3": np.zeros((1, 10), dtype=float),
                "A3": np.zeros((1, 10), dtype=float)
                }
        # array to store gradients
        self.grads = {
                "dW1": np.zeros((28 * 28, h1), dtype=float),
                "db1": np.zeros((1, h1), dtype=float),
                "dW2": np.zeros((h1, h2), dtype=float),
                "db2": np.zeros((1, h2), dtype=float),
                "dW3": np.zeros((h2, 10), dtype=float),
                "db3": np.zeros((1, 10), dtype=float)
                }
        # initial learning rate
        self.lr = 0.0005
        # initial beta parameter for momentum
        self.beta = 0.9
    
    def forward(self, x):
        """Forward-pass function
        Args:
            x.shape = (batch size, 28 * 28)
        Return:
            logits.shape = (batch size, 10)
        """
        N = x.shape[0]
        self.cache["X"] = x
        
        cur_x = x[0]
        Z1 = np.dot(cur_x, self.weights["W1"]) + self.weights["b1"]
        A1 = ReLU(Z1)
        Z2 = np.dot(A1, self.weights["W2"]) + self.weights["b2"]
        A2 = ReLU(Z2)
        Z3 = np.dot(A2, self.weights["W3"]) + self.weights["b3"]
        A3 = Softmax(Z3)
        self.cache["Z1"] = Z1
        self.cache["A1"] = A1
        self.cache["Z2"] = Z2
        self.cache["A2"] = A2
        self.cache["Z3"] = Z3
        self.cache["A3"] = A3

        for i in range(1, N):
            # iterate data in a batch
            cur_x = x[i]
            Z1 = np.dot(cur_x, self.weights["W1"]) + self.weights["b1"]
            A1 = ReLU(Z1)
            Z2 = np.dot(A1, self.weights["W2"]) + self.weights["b2"]
            A2 = ReLU(Z2)
            Z3 = np.dot(A2, self.weights["W3"]) + self.weights["b3"]
            A3 = Softmax(Z3)
            
            self.cache["Z1"] = np.concatenate((self.cache["Z1"], Z1), axis=0)
            self.cache["A1"] = np.concatenate((self.cache["A1"], A1), axis=0)
            self.cache["Z2"] = np.concatenate((self.cache["Z2"], Z2), axis=0)
            self.cache["A2"] = np.concatenate((self.cache["A2"], A2), axis=0)
            self.cache["Z3"] = np.concatenate((self.cache["Z3"], Z3), axis=0)
            self.cache["A3"] = np.concatenate((self.cache["A3"], A3), axis=0)
        
        logits = self.cache["A3"]
        return logits

    def backward(self, logits, labels):
        """Bachward-pass function
        Args:
            logits.shape = (batch size, 10)
            labels.shape = (batch size, 1)
        Return:
            None
        """
        N = labels.shape[0]
        
        # encode labels to one-hot format
        labels = one_hot_encoding(labels)
        
        dW1 = db1 = dW2 = db2 = dW3 = db3 = 0.0
        for i in range(N):
            # iterate data in a batch
            cur_logits = logits[i]
            cur_labels = labels[i]
            
            cur_dZ3 = cur_logits - cur_labels
            cur_dW3 = (1. / N) * np.dot(self.cache["A2"][i].reshape(-1, 1), cur_dZ3.reshape(1, -1))
            cur_db3 = (1. / N) * cur_dZ3.reshape(1, -1)

            cur_dA2 = np.dot(self.weights["W3"], cur_dZ3)
            cur_dZ2 = ReLU_back(cur_dA2, self.cache["Z2"][i])
            cur_dW2 = np.dot(self.cache["A1"][i].reshape(-1, 1), cur_dZ2.reshape(1, -1))
            cur_db2 = cur_dZ2.reshape(1, -1)
            
            cur_dA1 = np.dot(self.weights["W2"], cur_dZ2)
            cur_dZ1 = ReLU_back(cur_dA1, self.cache["Z1"][i])
            cur_dW1 = (1. / N) * np.dot(self.cache["X"][i].reshape(-1, 1), cur_dZ1.reshape(1, -1))
            cur_db1 = (1. / N) * cur_dZ1.reshape(1, -1)
            
            # accumulate gradients over whole batch
            dW1 += cur_dW1
            db1 += cur_db1
            dW2 += cur_dW2
            db2 += cur_db2
            dW3 += cur_dW3
            db3 += cur_db3
        
        # use momentum optimizer
        self.grads["dW1"] = (1. - self.beta) * dW1 + self.beta * self.grads["dW1"]
        self.grads["db1"] = (1. - self.beta) * db1 + self.beta * self.grads["db1"]
        self.grads["dW2"] = (1. - self.beta) * dW2 + self.beta * self.grads["dW2"]
        self.grads["db2"] = (1. - self.beta) * db2 + self.beta * self.grads["db2"]
        self.grads["dW3"] = (1. - self.beta) * dW3 + self.beta * self.grads["dW3"]
        self.grads["db3"] = (1. - self.beta) * db3 + self.beta * self.grads["db3"]


    def step(self):
        """update weights function
        Args:
            None
        Return:
            None
        """
        self.weights["W1"] -= self.lr * self.grads["dW1"]
        self.weights["b1"] -= self.lr * self.grads["db1"]
        self.weights["W2"] -= self.lr * self.grads["dW2"]
        self.weights["b2"] -= self.lr * self.grads["db2"]
        self.weights["W3"] -= self.lr * self.grads["dW3"]
        self.weights["b3"] -= self.lr * self.grads["db3"]
    
    def lr_decay(self):
        """decay model learning rate
        Args:
            None
        Return:
            None
        """
        self.lr /= 2
        
    def predict(self, x):
        """predict a category given input data
        Args:
            x.shape = (batch size, 28 * 28)
        Return:
            preds.shape = (batch, 1)
        """
        N = x.shape[0]
        
        preds = np.zeros((N, 1))
        logits = self.forward(x)
        for i in range(N):
            preds[i] = np.argmax(logits[i])
        
        return preds
    
    def CEloss(self, logits, labels):
        """compute cross entropy (CE) loss
        Args:
            logits.shape = (batch size, 10)
            labels.shape = (batch size, 1)
        Return:
            loss.shape = (1)
        """
        N = labels.shape[0]
        labels = one_hot_encoding(labels)
        log_sum = np.sum(np.multiply(labels, np.log(logits)))
        CE_loss = - (1. / N) * log_sum

        return CE_loss

    def save(self, path_to_checkpoints, tag):
        """function to save model weights
        Args:
            path_to_checkpoints: directory to save model weights
            tag: tag about current model weights
        Return:
            None
        """
        filename = os.path.join(path_to_checkpoints, "model_" + str(tag))
        np.save(filename, self.weights)

    def load(self, path_to_checkpoint):
        """function to load model weights
        Args:
            path_to_checkpoint: path to target model weights
        Return:
            None
        """
        checkpoint = np.load(path_to_checkpoint)
        for key in self.weights.keys():
            self.weights[key] = checkpoint[()][key]

