import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (L1 smooth variant).
    Loss = sqrt((pred - target)^2 + eps^2)
    """
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + (self.eps * self.eps))
        return torch.mean(loss)


class SSIMLoss(nn.Module):
    """
    Differentiable 2D Structural Similarity (SSIM) Loss for PyTorch Tensors.
    Loss = 1.0 - SSIM(pred, target)
    """
    def __init__(self, window_size=11, channel=1, data_range=1.0):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.data_range = data_range
        self.window = self.create_gaussian_window(window_size, channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.exp(torch.tensor([-(x - window_size // 2) ** 2 / float(2 * sigma ** 2) for x in range(window_size)]))
        return gauss / gauss.sum()

    def create_gaussian_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2):
        channel = img1.size(1)
        if self.window.device != img1.device or self.window.dtype != img1.dtype:
            self.window = self.window.to(device=img1.device, dtype=img1.dtype)

        window = self.window
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return ssim_map.mean()

    def forward(self, pred, target):
        return 1.0 - self.ssim(pred, target)


class CombinedLoss(nn.Module):
    """
    Combined Loss = CharbonnierLoss + 0.2 * SSIMLoss
    """
    def __init__(self, charbonnier_eps=1e-3, ssim_window_size=11, ssim_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.charbonnier = CharbonnierLoss(eps=charbonnier_eps)
        self.ssim_loss = SSIMLoss(window_size=ssim_window_size)
        self.ssim_weight = ssim_weight

    def forward(self, pred, target):
        l_char = self.charbonnier(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        total_loss = l_char + self.ssim_weight * l_ssim
        return total_loss, l_char, l_ssim
