import numpy as np

def accuracy(pred, labels):
    """
    Thi is example code.
    Customize your self.
    """
    pred = np.argmax(pred, axis=1)
    return np.sum(pred==labels)/float(labels.size)