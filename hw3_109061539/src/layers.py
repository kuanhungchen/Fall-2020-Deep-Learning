import numpy as np


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

        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding = padding

        # initialization
        self.weights = np.random.randn(*self.kernel_shape) * 0.1
        self.bias = np.random.randn(self.kernel_shape[3]) * 0.1

        self.grad = {'dW': None, 'db': None}

        self.update_required = True

    def forward(self, input_feat):
        """2D convolution operation
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

        B, iH, iW, _ = input_feat.shape
        kH, kW, _, oC = self.kernel_shape

        oH = int(((iH + 2 * self.padding - kH) // self.stride) + 1)
        oW = int(((iW + 2 * self.padding - kW) // self.stride) + 1)
        output_feat = np.zeros((B, oH, oW, oC))

        input_feat = self.get_padded_feat(input_feat)
        _, piH, piW, _ = input_feat.shape

        for b in range(B):
            h = 0
            for i in range(oH):
                w = 0
                for j in range(oW):
                    for oc in range(oC):
                        partial_kernel = np.reshape(self.weights[:, :, :, oc], (1, kH, kW, -1))
                        partial_image = input_feat[b, h:h + kH, w:w + kW, :]

                        output_feat[b, i, j, oc] = np.sum(partial_kernel * partial_image)
                    w += self.stride
                h += self.stride
        
        return output_feat + self.bias

    def backward(self, input_feat, upstream_grad):
        # TODO: compute grad and store in self.grads
        _, oH, oW, _ = d.shape

        raise NotImplementedError

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
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.randn(self.input_shape)
        self.bias = np.random.randn(self.output_shape)
        
        self.grads = {'dW': None, 'db': None}

        self.update_required = True

    def forward(self, input_feat):
        output_feat = np.dot(input_feat, self.weights) + self.bias
        return output_feat

    def backward(self, input_feat, upstream_grad):
       local_grad = np.dot(upstream_grad, self.weights.T)

       weights_grad = np.dot(input_feat, local_grad)
       bias_grad = local_grad.mean(axis=0) * input_feat.shape[0]

       self.grads['dW'] = weights_grad
       self.grads['db'] = bias_grad

       downstream_grad = local_grad
       return downstream_grad
    
    def update(self, learning_rate):
       self.weights = self.weights - learning_rate * self.grads['dW']
       self.bias = self.bias - learning_rate * self.grads['db']


class ReLU:
    """ReLU function layer
    """
    def __init__(self):
        self.update_required = False

    def forward(self, input_feat):
        output_feat = np.maximum(input_feat, 0)
        return output_feat

    def backward(self, input_feat, upstream_grad):
        local_grad = input_feat > 0
        downstream_grad = upstream_gradeam * local_grad
        return downstream_grad
