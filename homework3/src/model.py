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
            Conv2D(kernel_shape=(5, 5, self.INPUT_CHANNEL, 8)),
            ReLU(),
            Conv2D(kernel_shape=(5, 5, 8, 8)),
            ReLU(),
            Flatten(),
            FullyConnected(input_shape=4608, output_shape=1024),
            ReLU(),
            FullyConnected(input_shape=1024, output_shape=256),
            ReLU(),
            FullyConnected(input_shape=256, output_shape=64),
            ReLU(),
            FullyConnected(input_shape=64, output_shape=self.NUM_CLASS),
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
        with open(os.path.join(path_to_checkpoints, str(tag), 'model_arch.txt'), 'w') as fp:
            for i, layer in enumerate(self.layers):
                if layer.update_required is True:
                    weights_filename = os.path.join(
                        path_to_checkpoints, str(tag),
                        layer.name + '_' + str(i + 1) + '_weights'
                    )
                    np.save(weights_filename, layer.weights)
                    # print('[Model] saveing file {}'.format(weights_filename))

                    bias_filename = os.path.join(
                        path_to_checkpoints, str(tag),
                        layer.name + '_' + str(i + 1) + '_bias'
                    )
                    np.save(bias_filename, layer.bias)
                    # print('[Model] saving file {}'.format(bias_filename))

                    fp.write('[ID] {:2d} | [Name] {:15} | [Weights] {:15} | [Bias] {:15}\n'.format(
                        i + 1, layer.name, str(layer.weights.shape), str(layer.bias.shape)))
                else:
                    fp.write('[ID] {:2d} | [Name] {:15}\n'.format(i + 1, layer.name))
        fp.close()

    def load(self, path_to_checkpoint):
        for filename in sorted(os.listdir(path_to_checkpoint)):
            if os.path.splitext(filename)[1] != '.npy': continue
            name, layer_id, weights_or_bias = os.path.splitext(filename)[0].split('_')
            if name == self.layers[int(layer_id) - 1].name and self.layers[int(layer_id) - 1].update_required is True:
                # print('[Model] loading file {}'.format(filename))
                if weights_or_bias == 'weights':
                    self.layers[int(layer_id) - 1].weights = np.load(os.path.join(path_to_checkpoint, filename))
                else:
                    self.layers[int(layer_id) - 1].bias = np.load(os.path.join(path_to_checkpoint, filename))
