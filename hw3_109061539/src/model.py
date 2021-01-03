import numpy as np

from src.dataset import Dataset
from src.layers import Softmax, ReLU, Conv2D, FullyConnected, Flatten


class Model:

    INPUT_CHANNEL = 1
    NUM_CLASS = 3

    def __init__(self):
        self.layers = [
            Conv2D(kernel_shape=(5, 5, self.INPUT_CHANNEL, 4)),
            ReLU(),
            Conv2D(kernel_shape=(7, 7, 4, 8)),
            ReLU(),
            Conv2D(kernel_shape=(5, 5, 8, 4)),
            ReLU(),
            Flatten(),
            FullyConnected(input_shape=1296, output_shape=640),
            ReLU(),
            FullyConnected(input_shape=640, output_shape=256),
            ReLU(),
            FullyConnected(input_shape=256, output_shape=64),
            ReLU(),
            FullyConnected(input_shape=64, output_shape=self.NUM_CLASS),
            Softmax()
        ]

        self.lr = 0.0005

    def forward(self, x):
        for layer in self.layers:
            print("[forward] {}".format(layer.name))
            x = layer.forward(x)

        return x

    def backward(self, d):
        """back propagation in reversed order
        Args:
            d: The activation from last layer, which means the differnece
            between predicted output and ground truth.
        """

        for layer in reversed(self.layers):
            print("[backward] {}".format(layer.name))
            d = layer.backward(upstream_grad=d)
    
    def update(self):
        """update weights for each layer if needed
        """
        for layer in self.layers:
            if layer.update_required is True:
                print("[update] {}".format(layer.name))
                layer.update(self.lr)

    def predict(self, x):
        logits = self.forward(x)
        B = logits.shape[0]

        prediction = np.zeros((B, 1), dtype=int)
        for idx in range(B):
            prediction[idx, 0] = np.argmax(logits[idx, :])

        return prediction

    def save(self, path_to_checkpoints_dir, tag):
        raise NotImplementedError

    def load(self, path_to_checkpoint):
        raise NotImplementedError


if __name__ == '__main__':
    dataset = Dataset('./data', mode='train')
    model = Model()

    img, lbl = dataset[0]
    img = np.reshape(img, (1, 32, 32, 1))

    logits = model.forward(img)
    print(logits)
    init_grad = logits - [[1, 0, 0]]
    model.backward(d=init_grad)
    model.update()
