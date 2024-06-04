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

import torch
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg.proximal import soft_thresholding
def soft_thresholding_new(data, t):
    s = soft_thresholding(data, t)
    return s
def Umin(pu,Umax):
    if pu <= Umax :
        return pu
    else:
        return Umax
def Frobenius (data):
    out = torch.norm(data)
    return out
def Tucker(data,rank):
    data = tl.tensor(data)
    T= tucker(data, rank)
    Tucker1 = tl.tucker_to_tensor(T)
    return Tucker1