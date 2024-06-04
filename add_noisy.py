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
import random
import numpy as np
import torch
import scipy.io

# Normalize operation
def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min)/(max-min)

def variable_to_cv2_image(varim):
	r"""Converts a torch.autograd.Variable to an OpenCV image

	Args:
		varim: a torch.autograd.Variable
	"""
	nchannels = varim.size()[1]

	if nchannels == 1:
		res = (varim.data.cpu().numpy()[0, 0, :]*255.).clip(0, 255).astype(np.uint8)
	else:
		raise Exception('Number of color channels not supported')
	return res

def add_noisy_function(img_Y):
	img_noisy = np.zeros_like(img_Y)
	for i in range(img_Y.shape[2]):
		ch = img_Y[:,:,i]
		img_noisy[:,:,i] =  add_one_img(ch)
	# Save images
	print('**********img_noisy_Y***************')
	scipy.io.savemat(r'img_noisy_Y.mat', {'img_noisy': img_noisy})

	return img_noisy
def add_one_img(ch):
	# Check if input exists and if it is RGB
	global noisyimg

	imorig = np.expand_dims(ch, 0)
	imorig = np.expand_dims(imorig, 0)

	# Handle odd sizes
	expanded_h = False
	expanded_w = False
	sh_im = imorig.shape
	if sh_im[2] % 2 == 1:
		expanded_h = True
		imorig = np.concatenate((imorig, \
								 imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

	if sh_im[3] % 2 == 1:
		expanded_w = True
		imorig = np.concatenate((imorig, \
								 imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)

	imorig = minmaxscaler(imorig)
	imorig = torch.Tensor(imorig)


	# Add noise (noise_sigma can be 0.05, 0.1, 0.15, 0.2)
	noise_sigma = 0.05

	# Add random noise (random noise_sigma vary in the interval
	# [0, 0.1], [0, 0.2], [0, 0.3] and [0, 0.4], respectively)
	#noise_sigma = 0.4*random.random()

	if 1:
		noise = torch.FloatTensor(imorig.size()).normal_(mean=0, std=noise_sigma)
		imnoisy = imorig + noise

	if expanded_h:
		imnoisy = imnoisy[:, :, :-1, :]

	if expanded_w:
		imnoisy = imnoisy[:, :, :, :-1]

	noisyimg = variable_to_cv2_image(imnoisy)
	return noisyimg


