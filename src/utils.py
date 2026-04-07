import torch
import torch.nn as nn
import numpy as np
from config import POS_WEIGHT, LAMBDA_DICE, DEVICE

def getMetrics(TP, TN, FP, FN):
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
    iou_1  = TP / (TP + FP + FN + 1e-8)
    iou_0 = TN / (TN + FP + FN + 1e-8)
    miou = (iou_0 + iou_1) / 2 
        
    oa = (TP + TN)/(TP + TN + FP + FN + 1e-8)
    
    return precision, recall, f1, iou_1, miou, oa

def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

criterion_ce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))

def seg_loss(logits, targets):
    return criterion_ce(logits, targets) + LAMBDA_DICE * dice_loss(logits, targets)

def cosine_similarity(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    
    return (torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm())).item()

def l2_distance(a, b):
    return (a.float() - b.float()).norm().item()

def mean_pixel_diff(a, b):
    a_mean = a.float().mean(dim=(1, 2))
    b_mean = b.float().mean(dim=(1, 2))
    
    return (a_mean - b_mean).abs().mean().item()

def histogram_intersection(a, b, bins=64):
    a_np = a.float().numpy()
    b_np = b.float().numpy()
    score = 0.0
    
    for c in range(a_np.shape[0]):
        a_hist, _ = np.histogram(a_np[c].ravel(), bins=bins, range=(0, 1))
        b_hist, _ = np.histogram(b_np[c].ravel(), bins=bins, range=(0, 1))
        a_hist = a_hist / (a_hist.sum() + 1e-8)
        b_hist = b_hist / (b_hist.sum() + 1e-8)
        score += np.minimum(a_hist, b_hist).sum()
        
    return score / a_np.shape[0]
