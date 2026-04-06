import torch

@torch.no_grad()
def compute_mean_amplitude(tgt_list):
    mean_amp = None

    for img in tgt_list:
        fft = torch.fft.fft2(img, dim=(-2, -1))
        amp = torch.abs(fft)
        amp_shift = torch.fft.fftshift(amp, dim=(-2, -1))
        
        if mean_amp is None:
            mean_amp = amp_shift
        else:
            mean_amp += amp_shift

    return mean_amp / len(tgt_list)

@torch.no_grad()
def fourier_adapt(src, tgt_amp_shift, beta):
    _, H, W = src.shape
    
    src_fft = torch.fft.fft2(src, dim=(-2, -1))
    src_amp, src_phase = torch.abs(src_fft), torch.angle(src_fft)
    
    src_amp_shift = torch.fft.fftshift(src_amp, dim=(-2, -1))
    
    b_h, b_w = int(beta * H), int(beta * W)
    c_h, c_w = H // 2, W // 2
    h1, h2 = c_h - b_h, c_h + b_h
    w1, w2 = c_w - b_w, c_w + b_w
    
    src_amp_shift[..., h1:h2, w1:w2] = tgt_amp_shift[..., h1:h2, w1:w2]
    
    src_amp_new = torch.fft.ifftshift(src_amp_shift, dim=(-2, -1))
    src_fft_new = src_amp_new * torch.exp(1j * src_phase)
    
    new_img = torch.fft.ifft2(src_fft_new, dim=(-2, -1)).real.clamp(0, 1)
    
    return new_img
