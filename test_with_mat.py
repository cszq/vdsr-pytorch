import argparse, os
import torch
import numpy as np
import time, math, glob
import scipy.io as sio

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model = torch.load('./checkpoint/model_epoch_50.pth', map_location=lambda storage, loc: storage)["model"]
model = model.to(device)


scales = [2,3,4]


image_list = glob.glob('./Set5_mat'+"/*.*")

for scale in scales:
    avg_psnr_predicted = 0.0
    avg_psnr_bicubic = 0.0
    avg_elapsed_time = 0.0
    count = 0.0
    for image_name in image_list:
        if ('x' + str(scale)) in image_name:
            count += 1
            print("Processing ", image_name)
            im_gt_y = sio.loadmat(image_name)['im_gt_y']
            im_b_y = sio.loadmat(image_name)['im_b_y']
            #print(type(im_gt_y))
            im_gt_y = im_gt_y.astype(float)
            im_b_y = im_b_y.astype(float)

            psnr_bicubic = PSNR(im_gt_y, im_b_y, shave_border=scale)

            avg_psnr_bicubic += psnr_bicubic

            im_input = im_b_y/255.
            im_b_y = torch.from_numpy(im_input).to(device)
            im_b_y = im_b_y.float()

            im_b_y = im_b_y.unsqueeze(0).unsqueeze(0)
            im_b_y = im_b_y.to(device)
            with torch.no_grad():
                start_time = time.time()
                im_h_y = model(im_b_y).clamp(0.0, 1.0)
                elapsed_time = time.time() - start_time
                avg_elapsed_time += elapsed_time

            im_h_y = im_h_y.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

            psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=scale)
            #print("psnr_bicubic:{:.2f}  psnr_predicted:{:.2f}".format(psnr_bicubic, psnr_predicted))
            avg_psnr_predicted += psnr_predicted

    print("Scale=", scale)
    print("PSNR_predicted=", avg_psnr_predicted/count)
    print("PSNR_bicubic=", avg_psnr_bicubic/count)
    print("It takes average {:.2f}s for processing".format(avg_elapsed_time/count))
