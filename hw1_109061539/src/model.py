import numpy as np
# np.seterr(invalid="raise")


def ReLU(x):
    return np.maximum(0, x)

def ReLU_back(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def Softmax(x):
    """
    Args:
        x.shape(B, 10)
    Return:
        y.shape(B, 10)
    """
    exps = np.exp(x - x.max())
    y = exps / (1e-1 + sum(exps))
    return y

def one_hot_encoding(labels):
    """
    Args:
        labels.shape(B, 1)
    Return
        output.shape(B, 10)
    """
    N = labels.shape[0]
    output = np.zeros((N, 10), dtype=np.float)
    for i in range(N):
        output[i][labels[i]] = 1.0
    return output


class Model:
    """
    X: (1, 784)
    W1: (784, h1), b1 : (1, h1)
    Z1: (1, h1), A1: (1, h1)
    W2: (h1, h2), b2: (1, h2)
    Z2: (1, h2), A2: (1, h2)
    W3: (h2, 10), b3: (1, 10)
    Z3: (1, 10), A3: (1, 10)
    """
    def __init__(self, h1=256, h2=32):
        self.weights = {
                'W1': np.random.randn(28 * 28, h1) * 0.1, "b1": np.random.randn(1, h1) * 0.1,
                'W2': np.random.randn(h1, h2) * 0.1, "b2": np.random.randn(1, h2) * 0.1,
                'W3': np.random.randn(h2, 10) * 0.1, "b3": np.random.randn(1, 10) * 0.1
                }
        self.cache = {
                "Z1": np.zeros((1, h1), dtype=float),
                "A1": np.zeros((1, h1), dtype=float),
                "Z2": np.zeros((1, h2), dtype=float),
                "A2": np.zeros((1, h2), dtype=float),
                "Z3": np.zeros((1, 10), dtype=float),
                "A3": np.zeros((1, 10), dtype=float)
                }
        self.grads = {
                "dW1": np.zeros((28 * 28, h1), dtype=float),
                "db1": np.zeros((1, h1), dtype=float),
                "dW2": np.zeros((h1, h2), dtype=float),
                "db2": np.zeros((1, h2), dtype=float),
                "dW3": np.zeros((h2, 10), dtype=float),
                "db3": np.zeros((1, 10), dtype=float)
                }
        self.lr = 0.001
    
    def forward(self, x):
        """
        Args:
            x.shape = (B, 28 * 28)
        Return:
            logits.shape = (B, 10)
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

        return self.cache["A3"]
    
    def backward(self, logits, labels):
        """
        Args:
            logits.shape(B, 10)
            labels.shape(B, 1)
        Return:
            None
        """
        N = labels.shape[0]

        labels = one_hot_encoding(labels)
        
        for key in self.grads.keys():
            self.grads[key] = 0

        for i in range(N):
            cur_logits = logits[i]
            cur_labels = labels[i]
            
            dZ3 = cur_labels - cur_logits
            dW3 = np.dot(self.cache["A2"][i].reshape(-1, 1), dZ3.reshape(1, -1))
            db3 = dZ3.reshape(1, -1)
    
            dA2 = np.dot(self.weights["W3"], dZ3)
            dZ2 = ReLU_back(dA2, self.cache["Z2"][i])
            dW2 = np.dot(self.cache["A1"][i].reshape(-1, 1), dZ2.reshape(1, -1))
            # print("dW2 ==>", dW2.shape)
            db2 = dZ2.reshape(1, -1)
            # print("db2 ==>", db2.shape)
            
            dA1 = np.dot(self.weights["W2"], dZ2)
            dZ1 = ReLU_back(dA1, self.cache["Z1"][i])
            dW1 = np.dot(self.cache["X"][i].reshape(-1, 1), dZ1.reshape(1, -1))
            # print("dW1 ==>", dW1.shape)
            db1 = dZ1.reshape(1, -1)
            # print("db1 ==>", db1.shape)
            
            self.grads["dW1"] += dW1
            self.grads["db1"] += db1
            self.grads["dW2"] += dW2
            self.grads["db2"] += db2
            self.grads["dW3"] += dW3
            self.grads["db3"] += db3

            # self.grads["dW1"] = np.add(self.grads["dW1"], dW1)
            # self.grads["db1"] = np.add(self.grads["db1"], db1)
            # self.grads["dW2"] = np.add(self.grads["dW2"], dW2)
            # self.grads["db2"] = np.add(self.grads["db2"], db2)

        self.grads["dW1"] /= N
        self.grads["db1"] /= N
        self.grads["dW2"] /= N
        self.grads["db2"] /= N
        self.grads["dW3"] /= N
        self.grads["db3"] /= N

    def step(self):
        self.weights["W1"] += self.lr * self.grads["dW1"]
        self.weights["b1"] += self.lr * self.grads["db1"]
        self.weights["W2"] += self.lr * self.grads["dW2"]
        self.weights["b2"] += self.lr * self.grads["db2"]
        self.weights["W3"] += self.lr * self.grads["dW3"]
        self.weights["b3"] += self.lr * self.grads["db3"]
    
    def predict(self, x):
        """
        Args:
            x.shape(B, 28 * 28)
        Return:
            preds.shape(B, 1)
        """
        N = x.shape[0]

        preds = np.zeros((N, 1))
        logits = self.forward(x)
        for i in range(N):
            preds[i] = np.argmax(logits[i])
        
        return preds
    
    def l2_loss(self, x, labels):
        """
        Args:
            x.shape(B, 28 * 28)
            labels.shape(B, 1)
        Return:
            loss.shape(1)
        """
        N = x.shape[0]

        preds = self.predict(x)
        self.loss = sum([(label - pred) ** 2 for label, pred in zip(labels, preds)])
        
        return self.loss

    def CEloss(self, logits, labels):
        """
        Args:
            logits.shape(B, 10)
            labels.shape(B, 10)
        Return:
            loss.shape(1)
        """
        # TODO
        n = labels.shape[1]

        
        loss = - 1 / n * (np.dot(labels, np.log(logits).T) + np.dot(1 - labels, np.log(1 - logits).T))
        # loss = np.squeeze(loss)
        
        self.loss = sum(loss[0][:])
        return self.loss

    def save(self, path_to_checkpoints, step):
        # TODO
        raise NotImplementedError

    def load(self, path_to_checkpoints):
        # TODO
        raise NotImplementedError


