# import torch

# def mse(img1, img2):
#     return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

# def psnr(img1, img2):
#     mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
#     return 20 * torch.log10(1.0 / torch.sqrt(mse))

import torch

def mse(img1, img2):
    return ((img1 - img2) ** 2).mean()

def psnr(img1, img2):
    mse_value = mse(img1, img2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse_value))

