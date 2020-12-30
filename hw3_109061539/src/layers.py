import numpy as np


class Softmax:
    """softmax function layer
    """
    def __init__(self):

        self.name = "Softmax"
        self.update_required = False

        self.probs = None
    
    def forward(self, input_feat):

        exps = np.exp(input_feat)
        self.probs = exps / np.sum(exps)

        return self.probs
    
    def backward(self, upstream_grad):
        return upstream_grad


class ReLU:
    """ReLU function layer
    """
    def __init__(self):

        self.name = "ReLU"
        self.update_required = False

        self.activated_feat = None

    def forward(self, input_feat):
        self.activated_feat = np.maximum(input_feat, 0)
        return self.activated_feat

    def backward(self, upstream_grad):
        # print("self.activated_feat.shape:", self.activated_feat.shape)
        # print("upstream_grad.shape:", upstream_grad.shape)
        downstream_grad = np.array(upstream_grad, copy=True)
        downstream_grad[self.activated_feat <= 0] = 0
        # local_grad = self.activated_feat > 0
        # downstream_grad = upstream_grad * local_grad
        return downstream_grad


class Conv2D:
    """2D convolution layer
    (ref: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
    """
    def __init__(self, kernel_shape, stride=1, padding=0):
        """
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

        self.name = "Conv2D"
        self.update_required = True

        # variable initialization
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding = padding

        self.input_feat = None
        self.grads = {'dW': None, 'db': None}
        self.weights = np.random.randn(*self.kernel_shape) * 0.1
        self.bias = np.random.randn(self.kernel_shape[3]) * 0.1


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

        for b in range(B):
            h = 0
            for i in range(oH):
                w = 0
                for j in range(oW):
                    for oc in range(oC):
                        partial_weights = np.reshape(self.weights[:, :, :, oc], (1, kH, kW, -1))
                        partial_input_feat = input_feat[b, h:h + kH, w:w + kW, :]

                        output_feat[b, i, j, oc] = np.sum(partial_weights * partial_input_feat)
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

        """

        B, iH, iW, iC = self.input_feat.shape
        _, oH, oW, oC = upstream_grad.shape
        kH, kW, _, oC = self.kernel_shape

        downstream_grad = np.zeros((B, iH, iW, iC))
        self.grads['dW'] = np.zeros(self.kernel_shape)
        self.grads['db'] = upstream_grad.sum(axis=(0, 1, 2))

        input_feat = self.get_padded_feat(self.input_feat)
        
        # compute downstream gradient, shape = (B, iH, iW, iC)
        for b in range(B):
            h = 0
            for i in range(oH):
                w = 0
                for j in range(oW):
                    partial_weights = np.reshape(self.weights[:, :, :, :], (-1, kH, kW, iC)) # 1, 10, 10, 16
                    partial_upstream_grad = np.reshape(upstream_grad[b, i:i + 1, j:j + 1, :], (-1, 1, 1, 1)) # 32, 1, 1, 1
                    a = np.sum(partial_weights * partial_upstream_grad, axis=0) # 10, 10, 16
                    downstream_grad[b, h:h + kH, w:w + kW, :] += a
                    
                    w += self.stride
                h += self.stride
        
        # compute local gradient, shape = (kH, kW, iC, oC)
        for oc in range(oC):
            h = 0
            for i in range(oH):
                w = 0
                for j in range(oW):
                    partial_input_feat = np.reshape(self.input_feat[:, h:h + kH, w:w + kW, :], (-1, kH, kW, iC)) # 24, 10, 10, 16
                    partial_upstream_grad = np.reshape(upstream_grad[:, i:i + 1, j:j + 1, oc], (-1, 1, 1, 1)) # 24, 1, 1, 1
                    a = np.sum(partial_input_feat * partial_upstream_grad, axis=0)
                    self.grads['dW'][:, :, :, oc] += a


                    w += self.stride
                h += self.stride

        self.grads['dW'] /= B
        self.grads['db'] /= B

        return downstream_grad[:, self.padding:self.padding + iH, self.padding:self.padding + iW, :]

        ########
        # h = 0
        # for i in range(oH):
            # w = 0
            # for j in range(oW):
                # a = np.sum(
                    # self.weights[np.newaxis, :, :, :, :] *
                    # upstream_grad[:, i:i + 1, j:j + 1, np.newaxis, :],
                    # axis=4
                # )
                # downstream_grad[:, h:h + kH, w:w + kW, :] += a
                # self.grads['dW'] += np.sum(
                    # input_feat[:, h:h + kH, w:w + kW, :, np.newaxis] *
                    # upstream_grad[:, i:i + 1, j:j + 1, np.newaxis, :],
                    # axis=0
                # )
                # w += self.stride
            # h += self.stride

        # self.grads['dW'] /= B

        # return downstream_grad[:, self.padding:self.padding + iH, self.padding:self.padding + iW, :]

    def update(self, learning_rate):
        # TODO: implement conv layer update
        raise NotImplementedError

    def get_padded_feat(self, input_feat):
        """get padded version feature map given the value of padding
        Args:
            input_feat:
                The input feature map.

            padding:
                The value of padding.

        Return:
            padded_feat:
                The output feature map which is padded.
        """

        if self.padding == 0:
            padded_feat = input_feat
        else:
            B, iH, iW, iC = input_feat.shape
            padded_feat = np.zeros((B, iH + self.padding * 2, iW + self.padding * 2, iC))
            for b in range(B):
                padded_feat[b, self.padding:-self.padding, self.padding:-self.padding, :] = input_feat[b:, :, :, :]

        return padded_feat


class FullyConnected:
    """fully connected layer
    """
    def __init__(self, input_shape, output_shape):

        self.name = "FullyConnected"
        self.update_required = True
        
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.input_feat = None
        self.weights = np.random.randn(self.output_shape, self.input_shape) * 0.1
        self.bias = np.random.randn(1, self.output_shape) * 0.1
        self.grads = {'dW': None, 'db': None}

    def forward(self, input_feat):
        self.input_feat = input_feat

        output_feat = np.dot(input_feat, self.weights.T) + self.bias
        
        return output_feat

    def backward(self, upstream_grad):
        B = self.input_feat.shape[0]

        self.grads['dW'] = np.dot(upstream_grad.T, self.input_feat) / B
        self.grads['db'] = np.sum(upstream_grad) / B
        
        downstream_grad = np.dot(upstream_grad, self.weights)

        return downstream_grad
    
    def update(self, learning_rate):
        self.weights = self.weights - learning_rate * self.grads['dW']
        self.bias = self.bias - learning_rate * self.grads['db']


class Flatten:
    """reshape the input to 1-dimension (as known as flatten)
    """
    def __init__(self, input_shape=None):

        self.name = "Flatten"
        self.update_required = False
        
        self.input_shape = input_shape

    def forward(self, input_feat):
        self.input_shape = input_feat.shape
        assert self.input_shape == input_feat.shape

        B = input_feat.shape[0]
        output_feat = np.reshape(input_feat, (B, -1))

        return output_feat
    
    def backward(self, upstream_grad):
        downstream_grad = np.reshape(upstream_grad, self.input_shape)

        return downstream_grad


class MaxPool2D:
    def __init__(self, kernel_shape, stride):

        self.name = "MaxPool2D"
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
