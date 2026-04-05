import torch
from pytorch_wavelets import DWTForward, DWTInverse
from config import DEVICE

wave, levels, mode = 'haar', 1, 'reflect'

dwt = DWTForward(J=levels, wave=wave, mode=mode).to(DEVICE)
idwt = DWTInverse(wave=wave, mode=mode).to(DEVICE)

@torch.no_grad()
def compute_mean_LL(tgt_list):
    mean_LL = None
    
    for img in tgt_list:
        LL, _ = dwt(img.unsqueeze(0))
        
        if mean_LL is None:
            mean_LL = LL
        else:
            mean_LL += LL
    
    return mean_LL / len(tgt_list)

@torch.no_grad()
def wavelet_adapt(src, tgt_LL, alpha):
    src_LL, src_H = dwt(src.unsqueeze(0))
    new_LL = (1 - alpha) * src_LL + alpha * tgt_LL
    
    new_img = idwt((new_LL, src_H)).clamp(0, 1).squeeze(0)
    
    return new_img
