import torch
import torch.nn as nn

def loss_fn(outputs, labels):
    """
    Thi is example code.
    Customize your self.
    """
    return nn.CrossEntropyLoss()(outputs, labels)
    # return CrossEntropyLoss2d()(outputs, labels)

class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss()

    def forward(self, inputs, targets):
        return self.nll_loss(nn.functional.log_softmax(inputs, dim=1), targets.squeeze())