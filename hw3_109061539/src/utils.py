import numpy as np


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
