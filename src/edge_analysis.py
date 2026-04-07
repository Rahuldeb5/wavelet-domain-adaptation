import torch
import torch.nn.functional as F
from wavelet import dwt, wavelet_adapt
from fourier import fourier_adapt

def sobel_edges(img):
    gray = img.mean(dim=0, keepdim=True).unsqueeze(0)
    
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3).to(img.device)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3).to(img.device)
    
    edges_x = F.conv2d(gray, sobel_x, padding=1)
    edges_y = F.conv2d(gray, sobel_y, padding=1)
    
    return torch.sqrt(edges_x**2 + edges_y**2).squeeze()

def hf_energy(hf_subbands):
    return sum(h.pow(2).sum().item() for h in hf_subbands)

def edge_map_similarity(edges_orig, edges_adapted):
    a = edges_orig.flatten().float()
    b = edges_adapted.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-8)).item()

def subband_correlation(hf_orig, hf_adapted):
    names = ['LH', 'HL', 'HH']
    correlations = {}
    
    for i, name in enumerate(names):
        orig = hf_orig[0][0, :, i, :, :].flatten().float()
        adapt = hf_adapted[0][0, :, i, :, :].flatten().float()
        
        orig_c = orig - orig.mean()
        adapt_c = adapt - adapt.mean()
        
        r = (orig_c * adapt_c).sum() / (orig_c.norm() * adapt_c.norm() + 1e-8)
        correlations[name] = r.item()
    
    return correlations

@torch.no_grad()
def analyze_pair(src_tensor, tgt_mean_ll, tgt_mean_amp, alpha, beta):
    edges_orig = sobel_edges(src_tensor)
    _, src_hf = dwt(src_tensor.unsqueeze(0))
    orig_hf_e = hf_energy(src_hf)
    
    wav_img = wavelet_adapt(src_tensor, tgt_mean_ll, alpha)
    edges_wav = sobel_edges(wav_img)
    _, wav_hf = dwt(wav_img.unsqueeze(0))
    wav_sub = subband_correlation(src_hf, wav_hf) # should be close to 1.0
    
    fda_img = fourier_adapt(src_tensor, tgt_mean_amp, beta)
    edges_fda = sobel_edges(fda_img)
    _, fda_hf = dwt(fda_img.unsqueeze(0))
    fda_sub = subband_correlation(src_hf, fda_hf)
    
    return {
        'wav_hf_ratio': hf_energy(wav_hf) / (orig_hf_e + 1e-8),
        'fda_hf_ratio': hf_energy(fda_hf) / (orig_hf_e + 1e-8),
        'wav_edge_sim': edge_map_similarity(edges_orig, edges_wav),
        'fda_edge_sim': edge_map_similarity(edges_orig, edges_fda),
        'wav_LH_corr': wav_sub['LH'],
        'wav_HL_corr': wav_sub['HL'],
        'wav_HH_corr': wav_sub['HH'],
        'fda_LH_corr': fda_sub['LH'],
        'fda_HL_corr': fda_sub['HL'],
        'fda_HH_corr': fda_sub['HH'],
    }
