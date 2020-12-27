import numpy as np

from src.layers import Conv2D, ReLU


class Model:
    def __init__(self):
        self.layers = [
            Conv2D(kernel_shape=(3, 3, 3, 4)),
            ReLU(),
            Conv2D(kernel_shape=(3, 3, 4, 8)),
            ReLU(),
            Conv2D(kernel_shape=(3, 3, 8, 4))
        ]
        
        self.lr = 3e-4
    
    def forward(self, x):
        print("init shape ==>", x.shape)
        for layer in self.layers:
            x = layer.forward(x)
            print("shape ==>", x.shape)

        return x
    
    def backward(self, d):
        """back propagation in reversed order
        Args:
            d: The activation from last layer, which means the differnece
            between predicted output and ground truth.

        """

        for layer in reversed(self.layers):
            d = layer.backward(upstream_grad=d)
    
    def update(self):
        """update weights for each layer if needed
        """
        for layer in self.layers:
            if layer.update_required is True:
                layer.update(self.lr)


if __name__ == '__main__':
    model = Model()
    B = 1
    iC = 3
    test_image = np.zeros((B, 7, 7, iC))
    for b in range(B):
        for ic in range(iC):
            test_image[b, :, :, ic] = np.array(([
                [2, 3, 7, 4, 6, 2, 9],
                [6, 6, 9, 8, 7, 4, 3],
                [3, 4, 8, 3, 8, 9, 7],
                [7, 8, 3, 6, 6, 3, 4],
                [4, 2, 1, 8, 3, 4, 6],
                [3, 2, 4, 1, 9, 8, 3],
                [0, 1, 3, 9, 2, 1, 4]
            ]))

    logits = model.forward(test_image)
    print("logits.shape:", logits.shape)
    print("logits =", logits)
