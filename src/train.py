import torch
import torch.nn.functional as F
from utils import seg_loss, getMetrics
from config import EPOCHS, THRESHOLD, PATIENCE, DEVICE

import copy

def train_model(model, optimizer, train_loader, val_loader, src_region, tgt_region=None):
    best_iou = 0.0
    counter = 0
    
    best_state = copy.deepcopy(model.state_dict())
    
    print(f'src_region, tgt_region, epoch, train_loss, val_loss, val_iou')

    for epoch in range(EPOCHS):
        ### TRAIN ###
        model.train()
        running_loss, n_samples = 0.0, 0

        for images, masks in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            loss = seg_loss(logits, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * masks.numel()
            n_samples += masks.numel()
        
        training_loss = running_loss / max(1, n_samples)

        ### EVAL ###
        model.eval()
        val_loss, val_num = 0.0, 0
        TP, FP, FN, TN = 0,0,0,0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True).float()

                logits = model(images)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

                loss = seg_loss(logits, masks)
                val_loss += loss.item()

                preds = torch.sigmoid(logits) > THRESHOLD

                preds_flat = preds.view(-1)
                mask_flat = masks.view(-1)

                TP += ((preds_flat == 1) & (mask_flat == 1)).sum().item()
                FP += ((preds_flat == 1) & (mask_flat == 0)).sum().item()
                FN += ((preds_flat == 0) & (mask_flat == 1)).sum().item()
                TN += ((preds_flat == 0) & (mask_flat == 0)).sum().item()
                
                val_num += 1
             
        _, _, _, iou, _, _ = getMetrics(TP, TN, FP, FN)

        print(f'{src_region}, {tgt_region}, {epoch+1}, {training_loss :.3f}, {val_loss / val_num :.3f}, {iou:.4f}')

        if iou > best_iou:
            best_iou = iou
            counter = 0
            best_state = copy.deepcopy(model.state_dict())
        elif iou != 0.0:
            counter += 1
            if counter >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model

def test_model(model, test_loader):
    TP, FP, FN, TN = 0,0,0,0

    model.eval()
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True).float()

            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            preds = torch.sigmoid(logits) > THRESHOLD

            preds_flat = preds.view(-1)
            mask_flat = masks.view(-1)

            TP += ((preds_flat == 1) & (mask_flat == 1)).sum().item()
            FP += ((preds_flat == 1) & (mask_flat == 0)).sum().item()
            FN += ((preds_flat == 0) & (mask_flat == 1)).sum().item()
            TN += ((preds_flat == 0) & (mask_flat == 0)).sum().item()

    return getMetrics(TP, TN, FP, FN)

