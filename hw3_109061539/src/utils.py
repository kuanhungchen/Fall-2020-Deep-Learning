import numpy as np


def softmax(xs):
    exps = np.exp(xs)
    probs = exps / np.sum(exps)

    return probs


def CEloss(probs, labels):
    """cross entropy function
    Args:
        probs: The predicted probabilities of each class.
        labels: The corresponding ground truth.
    Return:
        CE_loss: The output CE loss.
    """
    
    N = probs.shape[0]
    labels = one_hot_encoding(labels)
    log_sum = np.sum(np.multiply(labels, np.log(probs)))
    CE_loss = - (1. / N) * log_sum

    return CE_loss


def one_hot_encoding(labels, num_class=3):
    """helper function to apply one hot encoding
    """
    
    N = len(labels)
    output = np.zeros((N, num_class), dtype=np.int32)
    for i in range(N):
        output[i][int(labels[i])] = 1

    return output


def maxPool2D(input_feat, kernel_size=2, stride=2):
    """2D max pooling operation
    (ref: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)

    Args:
        input_feat:
            The input feature map.

        kernel_size:
            The kernel size of max-pooling.
            Currently only supports for single integer.

        stride:
            The stride of max-pooling.
            Current only supports for single integer.
    
    Return:
        output_feat:
            The output feature map.
    """
    
    # TODO
    raise NotImplementedError
