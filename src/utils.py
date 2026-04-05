import torch
import torch.nn as nn
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
