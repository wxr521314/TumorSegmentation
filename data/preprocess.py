# encoding: utf-8
"""
 @project:TumorSegmentation
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 12/25/19 4:51 PM
 @desc: 构造肝脏分割网络的训练集，主要进行以下操作：
        1. 将 segmentation.nii 文件中的1,2全部用1表示（0表示背景，1表示肝脏，2表示肿瘤）
        2. 在 volume.nii 文件中，找到第一个和最后一个包含肝脏区域的切片，将它们的下标记作start_slice和end_slice
        3. 计算切片需要进行分块的范围 [ max(0,start_slice-20),  min(seg_array.shape[0], end_slice+20) ]
        4. 开始分块，默认设置块大小为48(深度)，步长为3，即每隔3个切片，生成一个数据块保存下来，格式为 .nii
"""

import os
import shutil
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from config.configuration import DefaultConfig
from tqdm import tqdm

opt = DefaultConfig()

# 预处理之前，清空之前处理的数据
if os.path.exists(opt.train_data_root):
    shutil.rmtree(opt.train_data_root)

os.mkdir(opt.train_data_root)
os.mkdir(opt.train_ct_root)
os.mkdir(opt.train_seg_root)


def generate_clock(image, array, start, end):
    """
    生成切片块
    :param image: 原始的volume.nii或segmentation.nii所对应的image
    :param array: 原始的volume.nii或segmentation.nii所对应的array
    :param start: 切片块的起始下标
    :param end:   切片块的终止下标
    :return: 返回切片块对应的image文件
    """
    array_block = array[start:end + 1, :, :] if end else array[start:, :, :]
    image_block = sitk.GetImageFromArray(array_block)
    image_block.SetDirection(image.GetDirection())
    image_block.SetOrigin(image.GetOrigin())
    image_block.SetSpacing(
        (image.GetSpacing()[0] * int(1 / opt.zoom_scale),
         image.GetSpacing()[1] * int(1 / opt.zoom_scale),
         opt.slice_thickness)
    )
    return image_block


def preprocess():
    volumes = [volume for volume in os.listdir(opt.origin_ct_root)]  # 所有volume.nii文件名组成的list

    idx = 0  # 切片块文件编号

    for ii, volume in tqdm(enumerate(volumes), total=len(volumes)):

        # 读取volume.nii文件
        liver = sitk.ReadImage(os.path.join(opt.origin_data_root, volume), sitk.sitkInt16)
        liver_array = sitk.GetArrayFromImage(liver)  # ndarray类型，shape为(切片数, 512, 512)

        # 读取segmentation.nii文件
        seg = sitk.ReadImage(
            os.path.join(opt.origin_seg_root, volume.replace('volume', 'segmentation')))
        seg_array = sitk.GetArrayFromImage(seg)
        seg_array[seg_array > 0] = 1  # 合并肝脏标签和肿瘤标签

        # 将灰度值在阈值之外的截断掉
        liver_array[liver_array > opt.gray_upper] = opt.gray_upper
        liver_array[liver_array < opt.gray_lower] = opt.gray_lower

        # 下采样原始array
        liver_array = ndimage.zoom(liver_array, (liver.GetSpacing()[-1] / opt.slice_thickness,
                                                 opt.zoom_scale,
                                                 opt.zoom_scale),
                                   order=3)  # shape变为(切片数//2,256,256)，采用三次插值
        seg_array = ndimage.zoom(seg_array, zoom=(seg.GetSpacing()[-1] / opt.slice_thickness, 1, 1),
                                 order=0)  # shape变为(切片数//2,512,512)，采用最近邻插值

        # 搜索切片分块区间
        z = np.any(seg_array, axis=(1, 2))  # 判断每一张切片中是否包含1（1表示肝脏），返回一个长度等于切片数的布尔数组
        start_slice, end_slice = np.where(z)[[0, -1]]  # np.where(z)返回数组中不为0的下标list
        start_slice = max(0, start_slice - opt.expand_slice)
        end_slice = min(seg_array.shape[0], end_slice + opt.expand_slice)
        if end_slice - start_slice < opt.block_size - 1:  # 过滤掉不足以生成一个切片块的原始样本
            continue
        liver_array = liver_array[start_slice, end_slice + 1, :, :]  # 截取原始CT影像中包含肝脏区间及拓张的所有切片
        seg_array = seg_array[start_slice, end_slice + 1, :, :]

        # 开始生成厚度为48的切片块，保存为nii格式
        l, r = 0, opt.block_size - 1
        while r < liver_array.shape[0]:
            # volume切片块和segmentation切片块生成
            liver_block = generate_clock(liver, liver_array, l, r)
            seg_block = generate_clock(seg, seg_array, l, r)

            liver_block_name = 'volume-' + str(idx) + '.nii'
            seg_block_name = 'segmentation-' + str(idx) + '.nii'
            sitk.WriteImage(liver_block, os.path.join(opt.train_ct_root, liver_block_name))
            sitk.WriteImage(seg_block, os.path.join(opt.train_seg_root, seg_block_name))

            idx += 1
            l += opt.stride
            r = l + opt.block_size - 1

        # 如果每隔opt.stride不能完整的将所有切片分块时，从后往前取到最后一个block
        if r != liver_array.shape[0] + opt.stride:
            # volume切片块生成
            liver_block = generate_clock(liver, liver_array, -opt.batch_size, None)
            seg_block = generate_clock(seg, seg_array, -opt.batch_size, None)

            liver_block_name = 'volume-' + str(idx) + '.nii'
            seg_block_name = 'segmentation-' + str(idx) + '.nii'
            sitk.WriteImage(liver_block, os.path.join(opt.train_ct_root, liver_block_name))
            sitk.WriteImage(seg_block, os.path.join(opt.train_seg_root, seg_block_name))

            idx += 1
