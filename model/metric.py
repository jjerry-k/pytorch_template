import numpy as np

def accuracy(pred, labels):
    """
    Thi is example code.
    Customize your self.
    """
    pred = np.argmax(pred, axis=1)
    return np.sum(pred==labels)/float(labels.size)

def iou(pred: np.array, labels: np.array):
    pred = pred.squeeze(1)
    
    intersection = (pred & labels).sum((1, 2))
    union = (pred | labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return thresholded  # Or thresholded.mean()