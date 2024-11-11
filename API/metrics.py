import torch
device = torch.device("cuda")

import numpy as np
from skimage.metrics import structural_similarity as cal_ssim

def MAE(pred, true):
    return np.mean(np.abs(pred-true),axis=(0,1)).sum()

def MSE(pred, true):
    return np.mean((pred-true)**2,axis=(0,1)).sum()

# cite the `PSNR` code from E3d-LSTM, Thanks!
# https://github.com/google/e3d_lstm/blob/master/src/trainer.py line 39-40
def PSNR(pred, true):
    mse = np.mean((np.uint8(pred * 255)-np.uint8(true * 255))**2)
    return 20 * np.log10(255) - 10 * np.log10(mse)

def metric(pred, true, mean, std, return_ssim_psnr=True, clip_range=[0, 1]):
    pred = pred*std + mean
    true = true*std + mean
    mae = MAE(pred, true)
    mse = MSE(pred, true)

    if return_ssim_psnr:
        pred = np.maximum(pred, clip_range[0])
        pred = np.minimum(pred, clip_range[1])
        ssim, psnr = 0.0, 0.0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                ssim += cal_ssim(pred[b, f].swapaxes(0, 2), true[b, f].swapaxes(0, 2), multichannel=True, channel_axis=2, data_range=1)
                psnr += PSNR(pred[b, f], true[b, f])
        ssim = ssim / (pred.shape[0] * pred.shape[1])
        psnr = psnr / (pred.shape[0] * pred.shape[1])
        # for b in range(pred.shape[0]):
        #     for f in range(pred.shape[1]):
        #         pred = pred[b,f]
        #         true = true[b,f]
        #         print(pred.shape, true.shape)
        #         # pred_np = pred.detach().cpu().numpy()
        #         # true_np = true.detach().cpu().numpy()
        #         psnr = PSNR(pred, true)
        #         # ssim = cal_ssim(pred.swapaxes(0, 2), true.swapaxes(0,2), multichannel=True, channel_axis=2, data_range=1)
        #         loss_psnr += psnr.item()
        #         loss_ssim += ssim.item()
        #         win_size = min(pred.shape[3], pred.shape[4])
        #         if win_size % 2 == 0:
        #             win_size -= 1
        #         ssim += cal_ssim(pred.swapaxes(0, 2), true.swapaxes(0, 2), multichannel=True, channel_axis=2)
        #         # psnr += PSNR(pred[b, f], true[b, f])
        # ssim = ssim / (pred.shape[0] * pred.shape[1])
        # psnr = psnr / (pred.shape[0] * pred.shape[1])
        return mse, mae, ssim, psnr
    else:
        return mse, mae