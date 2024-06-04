# coding=utf-8
"""
Denoise an image with the DPLRTA denoising method


@article{zeng2020hyperspectral,
  title={Hyperspectral image restoration via CNN denoiser prior regularized low-rank tensor recovery},
  author={Zeng, Haijin and Xie, Xiaozhen and Cui, Haojie and Zhao, Yuan and Ning, Jifeng},
  journal={Computer Vision and Image Understanding},
  volume={197},
  pages={103004},
  year={2020},
  publisher={Elsevier}
}

Copyright (C) 2021, Haijin Zeng <zeng_navy@163.com>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import argparse
import time
import numpy as np
import cv2
import torch
import h5py
import scipy.io
import torch.nn as nn
from torch.autograd import Variable
from models import FFDNet
from skimage.io import imread
from utils import batch_psnr, normalize, init_logger_ipol, \
    variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb
from add_noisy import add_noisy_function, minmaxscaler
from PNP_utils import soft_thresholding_new, Umin, Frobenius, Tucker
from skimage.measure.simple_metrics import compare_psnr




def main_finally(**args):
    # Init logger
    logger = init_logger_ipol()
    cuda_ = args['cuda']
    # add_noise = args['add_noise']
    # noise_sigma = args['noise_sigma']
    im_path = args['input']
    if im_path.endswith('.mat'):
        img = scipy.io.loadmat(im_path)
        img = img['OriData3'] * 255
    else:
        print("Please input .mat file!")

    img_noisy_Y = add_noisy_function(img)

    # If the noise image of the input image already exists,
    # you can directly input the noise image by uncommenting the following two lines of code

    # img_noisy_Y = scipy.io.loadmat('P_noisy_0.1_random.mat')
    # img_noisy_Y = img_noisy_Y['img_noisy']
    # Initialize parameters
    Z = S = N = T_1 = T_2 = T_3 = 0
    # The following parameters need to be manually adjusted in different cases to get the optimal results.
    U_max = 1000000

    p = 1500
    k = 0
    u = 100
    B = 0.00001
    t = 0.02
    # Assign an initial value to L. L_0.mat is the same shape as the input image
    L_0 = scipy.io.loadmat('L_0.mat')
    Z_0 = L_0['L_0']
    L_0 = L_0['L_0']
    e = 0.1e-6
    psnr_max = 0
    lamada = 0.01
    for k in range(10):
        temp = img_noisy_Y - S - N + Z + (T_1 - T_2) / u
        # The rank parameter need to be manually adjusted in different cases to get the optimal results
        rank = np.array([180, 180, 3])
        # 1th step :Tucker
        L = Tucker(temp, rank)
        L = torch.from_numpy(L)
        sigima = (t / u) ** 0.5
        temp_net = L + T_2 / u
        logger.info("\t The {0:0d} time".format(k))
        logger.info("\t sigima {0:0.4f} time".format(sigima))

        # 2th step
        Z = test_ffdnet(logger, cuda_, sigima, temp_net, img)
        if 1:
            psnr_new = 0
            img_new = img / 255.
            Z_new = Z / 255.
            for i in range(img.shape[2]):
                psnr_new += compare_psnr(img_new[:, :, i], Z_new[:, :, i], 1.)
            psnr_new = psnr_new / img.shape[2]
            if psnr_new > psnr_max:
                k_max = k
            logger.info("\t psnr_new {0:0.4f} ".format(psnr_new))

        if psnr_new > psnr_max:
            psnr_max = psnr_new
            scipy.io.savemat("Out_Z.mat", {'Out_Z': Z})
            logger.info("\t max  save {0:0d} ".format(k))
        Z = torch.from_numpy(Z)

        if not k:
            img_noisy_Y = torch.from_numpy(img_noisy_Y)
        temp = img_noisy_Y - L - N + T_1 / u
        temp_s = lamada / u

        # 3th step
        S = soft_thresholding_new(temp, temp_s)

        # 4th step
        temp = u * (img_noisy_Y - L - S) + T_1
        N = (temp / (u + 2 * B))

        T_1 = T_1 + u * (img_noisy_Y - L - S - N)

        # 5th step
        T_2 = T_2 + u * (L - Z)

        u = Umin((p * u), U_max)
        if not k:
            L_0 = torch.from_numpy(L_0)
            Z_0 = torch.from_numpy(Z_0)
        diff_L = L_0 - L
        diff_Z = Z_0 - Z
        temp1 = Frobenius(diff_L)
        temp2 = Frobenius(img_noisy_Y)
        temp3 = Frobenius(diff_Z)

        critical_value = (temp1 ** 2) / (temp2 ** 2)
        critical_Z = (temp3 ** 2) / (temp2 ** 2)
        if critical_value <= e:
            return L
        L_0 = L
        Z_0 = Z
        # logger.info("\tcritical_value {0:0.2f}".format(critical_value ))
        logger.info("\tcritical_Z {0:0.5f}".format(critical_Z))
    logger.info("\tPSNR_MAX {0:0.2f}".format(psnr_max))
    logger.info("\tK_MAX {0:0d}".format(k_max))


def test_ffdnet(logger, cuda_, noise_sigma, temp_net, img):
    img_out = np.zeros_like(temp_net)
    Runtime_avg = 0
    for i in range(temp_net.shape[2]):
        ch_t = temp_net[:, :, i]
        ch_img = img[:, :, i]
        img_out[:, :, i], Runtime = test_one_img(ch_t,  cuda_, noise_sigma, logger)
        Runtime_avg = Runtime_avg + Runtime
    Runtime_avg = Runtime_avg / img_out.shape[2]
    logger.info("\tRuntime {0:0.4f}s".format(Runtime_avg))
    return img_out


def test_one_img(ch_t, cuda_, noise_sigma, logger):
    # Check if input exists and if it is RGB
    global outimg
    try:
        rgb_den = is_rgb(ch_t)
    except:
        raise Exception('Could not open the input image')

    # Open image as a CxHxW torch.Tensor
    if rgb_den:
        in_ch = 3
        model_fn = 'models/net_rgb.pth'
    # from HxWxC to CxHxW, RGB image
    else:
        # from HxWxC to CxHxW grayscale image (C=1)
        in_ch = 1
        model_fn = 'models/net_gray.pth'
        ch_t = np.expand_dims(ch_t, 0)

    ch_t = np.expand_dims(ch_t, 0)
    ch_t = minmaxscaler(ch_t)
    ch_t = torch.Tensor(ch_t)

    # Absolute path to model file
    model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
                            model_fn)

    # Create model
    print('Loading model ...\n')
    net = FFDNet(num_input_channels=in_ch)

    # Load saved weights
    if cuda_:
        state_dict = torch.load(model_fn)
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
    else:
        state_dict = torch.load(model_fn, map_location='cpu')
        # CPU mode: remove the DataParallel wrapper
        state_dict = remove_dataparallel_wrapper(state_dict)
        model = net
    model.load_state_dict(state_dict)

    # Sets the model in evaluation mode (e.g. it removes BN)
    model.eval()

    # Sets data type according to CPU or GPU modes
    if cuda_:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    imnoisy = ch_t.clone()

    # Test mode
    with torch.no_grad():
        imnoisy = Variable(Variable(imnoisy.type(dtype)))

        nsigma = Variable(torch.FloatTensor([noise_sigma]).type(dtype))

        # Measure runtime return
        start_t = time.time()

        # Estimate noise and subtract it to the input image
        im_noise_estim = model(imnoisy, nsigma)
        outim = torch.clamp(imnoisy - im_noise_estim, 0., 1.)
        stop_t = time.time()

        time_ = stop_t - start_t

        outimg = variable_to_cv2_image(outim)
    return outimg, time_


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Parse arguments
    parser = argparse.ArgumentParser(description="DPLRTA_Test")
    parser.add_argument("--input", type=str, default="", \
                        help='path to input image')
    parser.add_argument("--suffix", type=str, default="", \
                        help='suffix to add to output name')
    parser.add_argument("--dont_save_results", action='store_true', \
                        help="don't save output images")
    parser.add_argument("--no_gpu", action='store_true', \
                        help="run model on CPU")
    argspar = parser.parse_args()

    # use CUDA?
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("\n### Testing DPLRTA model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main_finally(**vars(argspar))
