import torch

def AUROC(output, target):
    from sklearn.metrics import roc_auc_score
    with torch.no_grad():
        if output.dim() > 2:
            raise TypeError(f'Tensor with {output.dim()} dimensions is not available for AUROC')
        elif output.dim() == 2:
            if output.shape[-1] == 2:
                output = output[:, 1]
            else:
                raise TypeError(f'Only binary tensor is available for AUROC')
        else:
            pass

        y_score = output.numpy()
        y_true = target.numpy()

    return roc_auc_score(y_true, y_score)

def AUPRC(output, target):
    from sklearn.metrics import average_precision_score
    with torch.no_grad():
        if output.dim() > 2:
            raise TypeError(f'Tensor with {output.dim()} dimensions is not available for AUPRC')
        elif output.dim() == 2:
            if output.shape[-1] == 2:
                output = output[:, 1]
            else:
                raise TypeError(f'Only binary tensor is available for AUPRC')
        else:
            pass

        y_score = output.numpy()
        y_true = target.numpy()
        
    return average_precision_score(y_true, y_score)

def youden_j(output, target):
    from sklearn.metrics import roc_curve
    from numpy import argmax
    with torch.no_grad():
        y_score = output.numpy()
        y_true = target.numpy()

        fpr, tpr, thresholds = roc_curve(y_true, y_score)

    return thresholds[argmax(tpr - fpr)]

def accuracy(output, target):
    from sklearn.metrics import accuracy_score
    with torch.no_grad():
        if output.dim() > 2:
            raise TypeError(f'Tensor with {output.dim()} dimensions is not available for accuracy')
        elif output.dim() == 2:
            if output.shape[-1] == 2:
                output = output[:, 1]
                threshold = youden_j(output, target)
                y_score = (output > threshold).numpy().astype('int64')
            else:
                y_score = torch.argmax(output, dim=1).numpy()
        else:
            threshold = youden_j(output, target)
            y_score = (output > threshold).numpy().astype('int64')

        y_true = target.numpy()

    return accuracy_score(y_true, y_score)

def recall(output, target):
    from sklearn.metrics import recall_score
    with torch.no_grad():
        if output.dim() > 2:
            raise TypeError(f'Tensor with {output.dim()} dimensions is not available for recall')
        elif output.dim() == 2:
            if output.shape[-1] == 2:
                output = output[:, 1]
                threshold = youden_j(output, target)
                y_score = (output > threshold).numpy().astype('int64')
            else:
                y_score = torch.argmax(output, dim=1).numpy()
        else:
            threshold = youden_j(output, target)
            y_score = (output > threshold).numpy().astype('int64')

        y_true = target.numpy()

    return recall_score(y_true, y_score, zero_division=0)

def precision(output, target):
    from sklearn.metrics import precision_score
    with torch.no_grad():
        if output.dim() > 2:
            raise TypeError(f'Tensor with {output.dim()} dimensions is not available for precision')
        elif output.dim() == 2:
            if output.shape[-1] == 2:
                output = output[:, 1]
                threshold = youden_j(output, target)
                y_score = (output > threshold).numpy().astype('int64')
            else:
                y_score = torch.argmax(output, dim=1).numpy()
        else:
            threshold = youden_j(output, target)
            y_score = (output > threshold).numpy().astype('int64')

        y_true = target.numpy()

    return precision_score(y_true, y_score, zero_division=0)

def sensitivity(output, target):
    from sklearn.metrics import confusion_matrix
    with torch.no_grad():
        if output.dim() > 2:
            raise TypeError(f'Tensor with {output.dim()} dimensions is not available for f1')
        elif output.dim() == 2:
            if output.shape[-1] == 2:
                output = output[:, 1]
                threshold = youden_j(output, target)
                y_score = (output > threshold).numpy().astype('int64')
            else:
                y_score = torch.argmax(output, dim=1).numpy()
        else:
            threshold = youden_j(output, target)
            y_score = (output > threshold).numpy().astype('int64')

        y_true = target.numpy()

    tn, fp, fn, tp = confusion_matrix(y_true, y_score).ravel()

    if tp + fn == 0:
        return 0
    else:
        return tp / (tp + fn)

def specificity(output, target):
    from sklearn.metrics import confusion_matrix
    with torch.no_grad():
        if output.dim() > 2:
            raise TypeError(f'Tensor with {output.dim()} dimensions is not available for f1')
        elif output.dim() == 2:
            if output.shape[-1] == 2:
                output = output[:, 1]
                threshold = youden_j(output, target)
                y_score = (output > threshold).numpy().astype('int64')
            else:
                y_score = torch.argmax(output, dim=1).numpy()
        else:
            threshold = youden_j(output, target)
            y_score = (output > threshold).numpy().astype('int64')

        y_true = target.numpy()

    tn, fp, fn, tp = confusion_matrix(y_true, y_score).ravel()

    if tn + fp == 0:
        return 0
    else:
        return tn / (tn + fp)

def f1(output, target):
    from sklearn.metrics import f1_score
    with torch.no_grad():
        if output.dim() > 2:
            raise TypeError(f'Tensor with {output.dim()} dimensions is not available for f1')
        elif output.dim() == 2:
            if output.shape[-1] == 2:
                output = output[:, 1]
                threshold = youden_j(output, target)
                y_score = (output > threshold).numpy().astype('int64')
            else:
                y_score = torch.argmax(output, dim=1).numpy()
        else:
            threshold = youden_j(output, target)
            y_score = (output > threshold).numpy().astype('int64')

        y_true = target.numpy()

    return f1_score(y_true, y_score, zero_division=0)

def c_index(output, survival_time, vital_status):
    from lifelines.utils import concordance_index
    with torch.no_grad():
        if output.dim() > 2:
            raise TypeError(f'Tensor with {output.dim()} dimensions is not available for c-index')
        elif output.dim() == 2:
            if output.shape[-1] == 2:
                output = output[:, 1]
            else:
                raise TypeError(f'Only binary tensor is available for c-index')
        else:
            pass

        event_times = survival_time.numpy()
        predicted_scores = -output.numpy()
        event_observed = vital_status.numpy()
   

    try:
        cindex = concordance_index(event_times, predicted_scores, event_observed)
    except ZeroDivisionError:
        print("No admissible pairs for concordance index calculation.")
        print(f"vital_status: {vital_status}")
        print(f"vital_status unique values: {torch.unique(vital_status)}")
        print(f"survival_time: {survival_time}")
        cindex = 0.0  # Default value indicating no usable pairs



    return cindex
