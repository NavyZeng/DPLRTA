
# ABOUT

* Author    : Haijin Zeng <zeng_navy@163.com>


This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.


# OVERVIEW

This source code provides a PyTorch implementation of DPLRTA:
@article{zeng2020hyperspectral,
  title={Hyperspectral image restoration via CNN denoiser prior regularized low-rank tensor recovery},
  author={Zeng, Haijin and Xie, Xiaozhen and Cui, Haojie and Zhao, Yuan and Ning, Jifeng},
  journal={Computer Vision and Image Understanding},
  volume={197},
  pages={103004},
  year={2020},
  publisher={Elsevier}
}

# USER GUIDE

The code as is runs in Python 3.6 with the following dependencies:
## Dependencies
* [PyTorch v0.3.1](http://pytorch.org/)
* [scikit-image](http://scikit-image.org/)
* [tensorly](http://tensorly.org/stable/index.html)
* [torchvision](https://github.com/pytorch/vision)
* [OpenCV](https://pypi.org/project/opencv-python/)
* [HDF5](http://www.h5py.org/)
* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
## Usage

### 1. Testing

If you want to denoise an image using a one of the pretrained models
found under the *models* folder you can execute
```
python test_DPLRTA.py --input Pavia_80.mat  >run.txt
```
To run the algorithm on CPU instead of GPU:
```
python test_DPLRTA.py --input Pavia_80.mat  >run.txt --no_gpu
```
**NOTES**
* Models have been trained for values of noise in [0, 75]

### 2. Training

#### Prepare the databases

First, you will need to prepare the dataset composed of patches by executing
*prepare_patches.py* indicating the paths to the directories containing the 
training by passing *--trainset_dir*.

**NOTES**
* To prepare a grayscale dataset: ```python prepare_patches.py --gray```

#### Train a model

A model can be trained after having built the training databases
(i.e. *train_rgb.h5* for color denoising , and *train_gray.h5* for grayscale denoising ).
Only training on GPU is supported.
```
python train.py --gray
```
**NOTES**
* The training process can be monitored with TensorBoard as logs get saved
in the *log_dir* folder
* By default, models are trained for values of noise in [0, 75] (*--noiseIntL*
flag)
* A previous training can be resumed passing the *--resume_training* flag

# ABOUT THIS FILE

Copying and distribution of this file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved.  This file is offered as-is,
without any warranty.

# ACKNOLEDGMENTS

Some of the code is based on code by Kai Zhang <cskaizhang@gmail.com>