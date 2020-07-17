# encoding: utf-8
"""
 @project:TumorSegmenation
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 2020/2/18 20:49
 @desc: ct_array为ct影像，seg_array为对应的标签(0代表背景，1代表肝脏，2代表肿瘤)
"""

import os
from config.configuration import DefaultConfig
from torch.utils import data
import SimpleITK as sitk
import numpy as np
import torch

opt = DefaultConfig()


class CascadeData(data.Dataset):

    def __init__(self):
        self.volumes = [volume for volume in os.listdir(opt.train_data_root + '/ct')]

    def __getitem__(self, index):
        ct_name = self.volumes[index]  # 如 'volume-100.nii'
        seg_name = ct_name.replace('volume', 'segmentation')  # 如 'segmentation-100.nii'

        # 读入内存 ！！！注意，因为我们预处理文件时用了归一化处理，所以这里读文件时格式设为浮点数
        ct = sitk.ReadImage(os.path.join(opt.train_data_root + '/ct', ct_name), sitk.sitkFloat32)
        seg = sitk.ReadImage(os.path.join(opt.train_data_root + '/seg', seg_name), sitk.sitkInt8)
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        # 将array转换为tensor.Tensor类型
        ct_tensor = torch.as_tensor(ct_array).float().unsqueeze(0)  # shape 变为(1,48,256,256)，1表示通道数
        seg_tensor = torch.as_tensor(seg_array)

        return ct_tensor, seg_tensor

    def __len__(self):
        return len(self.volumes)
