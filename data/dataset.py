# encoding: utf-8
"""
 @project:TumorSegmentation
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 12/25/19 4:44 PM
 @desc: 对Dataset类进行封装，创建肝脏分割网络的训练集
"""
import os
from config.configuration import DefaultConfig
from torch.utils import data
import SimpleITK as sitk
import numpy as np
import torch

opt = DefaultConfig()


class Liver(data.Dataset):

    def __init__(self):
        self.volumes = [volume for volume in os.listdir(opt.train_data_root + '/ct')]

    def __getitem__(self, index):
        liver_name = self.volumes[index]  # 如 'volume-100.nii'
        seg_name = liver_name.replace('volume', 'segmentation')  # 如 'segmentation-100.nii'

        # 读入内存
        liver = sitk.ReadImage(os.path.join(opt.train_data_root + '/ct', liver_name), sitk.sitkInt16)
        seg = sitk.ReadImage(os.path.join(opt.train_data_root + '/seg', seg_name), sitk.sitkInt8)
        liver_array = sitk.GetArrayFromImage(liver)
        seg_array = sitk.GetArrayFromImage(seg)

        # 对切片块中的每一个切片，进行归一化操作
        liver_array = liver_array.astype(np.float32)
        liver_array = liver_array / 200

        # 将array转换为tensor.Tensor类型
        liver_tensor = torch.as_tensor(liver_array).float().unsqueeze(0)  # shape 变为(1,48,256,256)，1表示通道数
        seg_tensor = torch.as_tensor(seg_array)

        return liver_tensor, seg_tensor

    def __len__(self):
        return len(self.volumes)


class Tumor(data.Dataset):
    def __init__(self):
        self.volumes = [volume for volume in os.listdir(opt.train_data_root_2 + '/ct')]

    def __getitem__(self, index):
        tumor_name = self.volumes[index]  # 如 'volume-100.nii'
        seg_name = tumor_name.replace('volume', 'segmentation')  # 如 'segmentation-100.nii'

        # 读入内存，因为我们处理完图像进行了归一化操作，所以在读取图像时，需要设定为 sitk.sitkFloat32
        tumor = sitk.ReadImage(os.path.join(opt.train_data_root_2 + '/ct', tumor_name), sitk.sitkFloat32)
        seg = sitk.ReadImage(os.path.join(opt.train_data_root_2 + '/seg', seg_name), sitk.sitkInt8)
        tumor_array = sitk.GetArrayFromImage(tumor)
        seg_array = sitk.GetArrayFromImage(seg)

        # 将array转换为tensor.Tensor类型
        tumor_tensor = torch.as_tensor(tumor_array).float().unsqueeze(0)  # shape 变为(1,48,256,256)，1表示通道数
        seg_tensor = torch.as_tensor(seg_array)

        return tumor_tensor, seg_tensor

    def __len__(self):
        return len(self.volumes)
