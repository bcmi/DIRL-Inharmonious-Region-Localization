import numpy as np
import torch
from sklearn.metrics import average_precision_score	

def normPRED(d, eps=1e-2):
    ma = torch.max(d)
    mi = torch.min(d)

    if ma-mi<eps:
        dn = d-mi
    else:
        dn = (d-mi)/(ma-mi)

    return dn

def compute_mAP(outputs, labels):
    y_true = labels.cpu().detach().view(labels.size(0),-1).numpy()
    y_pred = outputs.cpu().detach().view(labels.size(0),-1).numpy()
    AP = []
    for i in range(y_true.shape[0]):
        AP.append(average_precision_score(y_true[i],y_pred[i]))
    return np.mean(AP)

def compute_IoU(pred, gt, threshold=0.5, eps=1e-6):
    pred = torch.where(pred > threshold, torch.ones_like(pred), torch.zeros_like(pred)).to(pred.device)
    intersection = (pred * gt).sum(dim=[1,2,3])
    union = pred.sum(dim=[1,2,3]) + gt.sum(dim=[1,2,3]) - intersection
    return (intersection / (union+eps)).mean().item()

def MAE(pred, gt):
    if isinstance(pred, torch.Tensor):
        return torch.mean(torch.abs(pred - gt))
    elif isinstance(pred, np.ndarray):
        return np.mean(np.abs(pred-gt))

def FScore(pred, gt, beta2=1.0, threshold=0.5, eps=1e-6, reduce_dims=[1,2,3]):
    if isinstance(pred, torch.Tensor):
        if threshold == -1: threshold = pred.mean().item() * 2
        ones = torch.ones_like(pred).to(pred.device)
        zeros = torch.zeros_like(pred).to(pred.device)
        pred_ = torch.where(pred > threshold, ones, zeros)
        gt = torch.where(gt>threshold, ones, zeros)
        total_num = pred.nelement()

        TP = (pred_ * gt).sum(dim=reduce_dims)
        NumPrecision = pred_.sum(dim=reduce_dims)
        NumRecall = gt.sum(dim=reduce_dims)
        
        precision = TP / (NumPrecision+eps)
        recall = TP / (NumRecall+eps)
        F_beta = (1+beta2)*(precision * recall) / (beta2*precision + recall + eps)
        F_beta = F_beta.mean()
        
    elif isinstance(pred, np.ndarray):
        if threshold == -1: threshold = pred.mean()* 2
        pred_ = np.where(pred > threshold, 1.0, 0.0)
        gt = np.where(gt > threshold, 1.0, 0.0)
        total_num = np.prod(pred_.shape)

        TP = (pred_ * gt).sum()
        NumPrecision = pred_.sum()
        NumRecall = gt.sum()
        
        precision = TP / (NumPrecision+eps)
        recall = TP / (NumRecall+eps)
        F_beta = (1+beta2)*(precision * recall) / (beta2*precision + recall + eps)

    return F_beta

if __name__ == "__main__":
    gt = torch.ones((1,1,3,3))    
    gt[0][0][1][1] = 0
    pred = torch.ones((1,1,3,3))    
    pred[0][0][1][2] = 0
    pred[0][0][1][0] = 0
    print(compute_IoU(pred, gt))