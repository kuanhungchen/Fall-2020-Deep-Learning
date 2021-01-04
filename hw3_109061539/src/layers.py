import numpy as np


class Softmax:
    """softmax function layer
    """
    def __init__(self):
        self.name = 'Softmax'
        self.update_required = False

        self.probs = None
    
    def forward(self, input_feat):
        """softmax function forward pass
        """
        exps = np.exp(input_feat)
        output_feat = self.probs = exps / np.sum(exps, axis=1, keepdims=True)

        return output_feat
    
    def backward(self, upstream_grad):
        """softmax function backward pass
        """
        return upstream_grad


class ReLU:
    """ReLU function layer
    """
    def __init__(self):
        self.name = 'ReLU'
        self.update_required = False

        self.activated_feat = None

    def forward(self, input_feat):
        """ReLU function forward pass
        """
        self.activated_feat = np.maximum(0, input_feat)
        return self.activated_feat

    def backward(self, upstream_grad):
        """ReLU function backward pass
        """
        downstream_grad = np.array(upstream_grad, copy=True)
        downstream_grad[self.activated_feat <= 0] = 0
        return downstream_grad


class Conv2D:
    def __init__(self, kernel_shape, stride=1, padding=0):
        """2D convolution layer
        (ref: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

        Args:
            kernel_shape:
                The shape (kH, kW, iC, oC) of the kernel in this layer,
                where kH, kW, iC, oC means kernel height, kernel width,
                input channel and output channel.

            stride:
                The stride of convolution operation, default is 1.

            padding:
                The padding of convolution operation, default is 0.
        """
        self.name = 'Conv2D'
        self.update_required = True

        # variable initialization
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding = padding
        self.input_feat = None
        self.weights = np.random.randn(*self.kernel_shape) * 0.1
        self.bias = np.random.randn(self.kernel_shape[3]) * 0.1
        self.grads = {'dW': None, 'db': None}

    def forward(self, input_feat):
        """2D convolution forward pass
        Args:
            input_feat:
                The input feature map or the image.
                The shape is (B, iH, iW, iC), where B, iH, iW, iC means
                batch_size, input height, input width and input channel.

        Returns:
            output_feat:
                The output feature map.
                The shape is (B, oH, oW, oC), where B, oH, oW, oC means
                batch_size, output height, output width and output channel.
        """
        self.input_feat = input_feat
        B, iH, iW, _ = input_feat.shape
        kH, kW, _, oC = self.kernel_shape

        oH = int(((iH + 2 * self.padding - kH) // self.stride) + 1)
        oW = int(((iW + 2 * self.padding - kW) // self.stride) + 1)
        output_feat = np.zeros((B, oH, oW, oC))

        input_feat = self.get_padded_feat(input_feat)
        h = 0
        for i in range(oH):
            w = 0
            for j in range(oW):
                output_feat[:, i, j, :] = np.sum(
                    input_feat[:, h:h + kH, w:w + kW, :, np.newaxis] *
                    self.weights[np.newaxis, :, :, :],
                    axis=(1, 2, 3)
                )
                w += self.stride
            h += self.stride

        return output_feat + self.bias

    def backward(self, upstream_grad):
        """2D convolution backward pass
        Args:
            upstream_grad:
                The upstream gradient.
                The shape is (B, oH, oW, oC), where B, oH, oW, oC means
                batch_size, output height, output width and output channel.

        Return:
            downstream_grad:
                The downstream gradient.
                The shape is (B, iH, iW, iC)
        """
        B, iH, iW, iC = self.input_feat.shape
        _, oH, oW, _ = upstream_grad.shape
        kH, kW, _, oC = self.kernel_shape
        input_feat = self.get_padded_feat(self.input_feat)
        downstream_grad = np.zeros_like(self.input_feat)
        self.grads['dW'] = np.zeros_like(self.weights)
        self.grads['db'] = upstream_grad.sum(axis=(0, 1, 2)) / B

        h = 0
        for i in range(oH):
            w = 0
            for j in range(oW):
                downstream_grad[:, h:h + kH, w:w + kW, :] += np.sum(
                    self.weights[np.newaxis, :, :, :, :] *
                    upstream_grad[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=4
                )
                self.grads['dW'] += np.sum(
                    self.input_feat[:, h:h + kH, w:w + kW, :, np.newaxis] *
                    upstream_grad[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=0
                )
                w += self.stride
            h += self.stride

        self.grads['dW'] /= B

        return downstream_grad[:, self.padding:self.padding + iH, self.padding:self.padding + iW, :]

    def update(self, learning_rate):
        """update weights and bias
        """
        self.weights = self.weights - learning_rate * self.grads['dW']
        self.bias = self.bias - learning_rate * self.grads['db']

    def get_padded_feat(self, input_feat):
        """get padded version feature map given the value of padding
        Args:
            input_feat:
                The input feature map.

            padding:
                The value of padding.j

        Return:
            padded_feat:
                The output feature map which is padded.
        """
        if self.padding == 0:
            padded_feat = input_feat
        else:
            B, iH, iW, iC = input_feat.shape
            padded_feat = np.zeros((B, iH + self.padding * 2, iW + self.padding * 2, iC))
            padded_feat[:, self.padding:self.padding + iH, self.padding:self.padding + iW, :] = input_feat

        return padded_feat


class FullyConnected:
    """fully connected layer
    """
    def __init__(self, input_shape, output_shape):
        self.name = 'FullyConnected'
        self.update_required = True
        
        # variable initialization
        self.input_feat = None
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.randn(self.output_shape, self.input_shape) * 0.1
        self.bias = np.random.randn(1, self.output_shape) * 0.1
        self.grads = {'dW': None, 'db': None}

    def forward(self, input_feat):
        """fully connected forward pass
        """
        self.input_feat = input_feat
        output_feat = np.dot(input_feat, self.weights.T) + self.bias
        
        return output_feat

    def backward(self, upstream_grad):
        """fully connected backward pass
        """
        B = self.input_feat.shape[0]

        self.grads['dW'] = np.dot(upstream_grad.T, self.input_feat) / B
        self.grads['db'] = np.sum(upstream_grad, axis=0, keepdims=True) / B
        downstream_grad = np.dot(upstream_grad, self.weights)

        return downstream_grad
    
    def update(self, learning_rate):
        """update weights and bias
        """
        self.weights = self.weights - learning_rate * self.grads['dW']
        self.bias = self.bias - learning_rate * self.grads['db']


class Flatten:
    """Flatten layer which reshapes the input to 1-dimension
    """
    def __init__(self):
        self.name = 'Flatten'
        self.update_required = False
        
        self.input_shape = None

    def forward(self, input_feat):
        """flatten layer forward pass
        """
        self.input_shape = input_feat.shape
        B = input_feat.shape[0]

        output_feat = np.reshape(input_feat, (B, -1))

        return output_feat
    
    def backward(self, upstream_grad):
        """flatten layer backward pass
        """
        downstream_grad = np.reshape(upstream_grad, self.input_shape)

        return downstream_grad


class MaxPool2D:
    def __init__(self, kernel_shape, stride):
        """2D max pooling operation
        (ref: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)

        Args:
            kernel_shape:
                The kernel shape of max-pooling.
                Currently only supports for single integer.

            stride:
                The stride of max-pooling.
                Current only supports for single integer.

        Return:
            output_feat:
                The output feature map.
        """

        self.name = 'MaxPool2D'
        self.update_required = False

        self.kernel_shape = kernel_shape
        self.stride = stride

    def forward(self, input_feat):
        # TODO
        raise NotImplementedError
        B, iH, iW, iC = input_feat.shape
        kH, kW = self.kernel_shape

        oH = int((iH - kH) // self.stride) + 1
        oW = int((iW - kW) // self.stride) + 1

        output_feat = np.zeros((B, oH, oW, iC))

        for b in range(B):
            h = 0
            for i in range(iH):
                w = 0
                for j in range(iW):
                    for ic in range(iC):
                        output_feat[b, i, j, ic] = np.maximum(input_feat[b, h:h + kH, w:w + kW, ic])
                    w += self.stride
                h += self.stride

        return output_feat
 
    def backward(self, upstream_grad):
        # TODO
        raise NotImplementedError
