import torch.nn as nn


# Model should include nn.LogSoftmax layer
def nll_loss(*args, **kwargs):
    return nn.NLLLoss(*args, **kwargs)

def cross_entropy(*args, **kwargs):
    return nn.CrossEntropyLoss(*args, **kwargs)

# Model should include nn.Sigmoid layer
def bce_loss(*args, **kwargs):
    return nn.BCELoss(*args, **kwargs)

def bce_with_logits_loss(*args, **kwargs):
    return nn.BCEWithLogitsLoss(*args, **kwargs)
