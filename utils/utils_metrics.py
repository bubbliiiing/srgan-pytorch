import torch
import torch.nn.functional as F
from math import exp
import numpy as np
 
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
 
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def SSIM(img1, img2, window_size=11, window=None, size_average=True, full=False):
    img1 = (img1 * 0.5 + 0.5) * 255
    img2 = (img2 * 0.5 + 0.5) * 255
    min_val = 0
    max_val = 255
    L = max_val - min_val
    img2 = torch.clamp(img2, 0.0, 255.0)
 
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
 
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
 
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
 
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
 
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
 
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
 
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
 
    if full:
        return ret, cs
    return ret
 
def tf_log10(x):
    numerator = torch.log(x)
    denominator = torch.log(torch.tensor(10.0))
    return numerator / denominator

def PSNR(img1, img2):
    img1 = (img1 * 0.5 + 0.5) * 255
    img2 = (img2 * 0.5 + 0.5) * 255
    max_pixel = 255.0
    img2 = torch.clamp(img2, 0.0, 255.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (torch.mean(torch.pow(img2 - img1, 2))))
