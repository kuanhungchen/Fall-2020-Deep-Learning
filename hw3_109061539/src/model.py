import os
import numpy as np

from src.dataset import Dataset
from src.layers import Softmax, ReLU, Conv2D, FullyConnected, Flatten


class Model:

    INPUT_CHANNEL = 1
    NUM_CLASS = 3

    def __init__(self, lr=3e-4):
        # definition of each layer of the model
        self.layers = [
            Conv2D(kernel_shape=(7, 7, self.INPUT_CHANNEL, 4)),
            ReLU(),
            Conv2D(kernel_shape=(7, 7, 4, 8)),
            ReLU(),
            Flatten(),
            FullyConnected(input_shape=3200, output_shape=400),
            ReLU(),
            FullyConnected(input_shape=400, output_shape=50),
            ReLU(),
            FullyConnected(input_shape=50, output_shape=self.NUM_CLASS),
            ReLU(),
            Softmax()
        ]

        # learning rate of optimizer (currently using SGD as optimizer)
        self.lr = lr

    def forward(self, x):
        """forward pass through each layer
        Args:
            x: The input images.
        Return:
            logits: The logits predicted by model.
        """
        for layer in self.layers:
            x = layer.forward(x)
            # print("[Model] {} forward ok".format(layer.name))

        logits = x

        return logits

    def backward(self, d):
        """back propagation in reversed order
        Args:
            d: The activation from last layer, which means the differnece
            between predicted output and ground truth.
        """
        for layer in reversed(self.layers):
            d = layer.backward(upstream_grad=d)
            # print("[Model] {} backward ok".format(layer.name))
    
    def update(self):
        """update weights for each layer if needed
        """
        for layer in self.layers:
            if layer.update_required is True:
                layer.update(self.lr)
                # print("[Model] {} update ok".format(layer.name))

    def save(self, path_to_checkpoints, tag):
        """save model weights into numpy files
        Args:
            path_to_checkpoints: The directory of saved model weights.
            tag: folder name to save the current model weights
        """
        os.makedirs(os.path.join(path_to_checkpoints, str(tag)), exist_ok=True)
        for i, layer in enumerate(self.layers):
            if layer.update_required is True:
                weights_filename = os.path.join(
                    path_to_checkpoints, str(tag),
                    layer.name + '_' + str(i + 1) + '_weights'
                )
                np.save(weights_filename, layer.weights)
                print('[Model] saveing file {}'.format(weights_filename))
                bias_filename = os.path.join(
                    path_to_checkpoints, str(tag),
                    layer.name + '_' + str(i + 1) + '_bias'
                )
                np.save(bias_filename, layer.bias)
                print('[Model] saving file {}'.format(bias_filename))

    def load(self, path_to_checkpoint):
        for filename in sorted(os.listdir(path_to_checkpoint)):
            name, layer_id, weights_or_bias = os.path.splitext(filename)[0].split('_')
            if name == self.layers[int(layer_id) - 1].name and self.layers[int(layer_id) - 1].update_required is True:
                print('[Model] loading file {}'.format(filename))
                if weights_or_bias == 'weights':
                    self.layers[int(layer_id) - 1].weights = np.load(os.path.join(path_to_checkpoint, filename))
                else:
                    self.layers[int(layer_id) - 1].bias = np.load(os.path.join(path_to_checkpoint, filename))
        

if __name__ == '__main__':
    model = Model()
    model.save('./checkpoints/', 'test')
