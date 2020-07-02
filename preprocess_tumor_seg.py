# encoding: utf-8
"""
 @project:TumorSegmentation
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 1/2/20 4:14 PM
 @desc: 为训练肿瘤分割模型而进行预处理，主要完成以下操作
        (1)从原始训练集中，读取volume.nii、segmentation.nii文件，存为ct_array、seg_array
        (2)将seg_array中的1变为0，2变为1，然后和ct_array进行element-wise的乘法，得到new_ct_array
        (3)接下来进行和处理肝脏分割训练集一样的操作：
            (3.1) 阈值截断
            (3.2) 插值法
            (3.3) 轴向扩张
            (3.4) new_ct_array和seg_array分块处理，并保存

        # Attention，因为我们处理完图像进行了归一化操作，所以在读取图像时，需要设定为 sitk.sitkFloat32
"""
import os
import pickle
import shutil
from config.configuration import DefaultConfig
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
import warnings
import scipy.ndimage as ndimage

warnings.filterwarnings("ignore")
opt = DefaultConfig()

# 预处理之前，清空之前处理的数据
if os.path.exists(opt.train_data_root_2):
    shutil.rmtree(opt.train_data_root_2)

os.mkdir(opt.train_data_root_2)
os.mkdir(opt.train_data_root_2 + '/ct')
os.mkdir(opt.train_data_root_2 + '/seg')


def generate_clock(image, array, start, end):
    """
    生成厚度为48的切片块
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


def get_train_list():
    with open('data/val_volumes_list.txt', 'rb') as f:
        val_list = pickle.load(f)
    all_list = os.listdir(opt.origin_train_root + '/ct')
    return [x for x in all_list if x not in val_list]  # 求差集


def preprocess():
    volumes = get_train_list()  # 获取训练集的文件名列表

    idx = 0  # 切片块文件编号

    for ii, volume in tqdm(enumerate(volumes), total=len(volumes)):
        # 读取volume.nii文件
        ct = sitk.ReadImage(os.path.join(opt.origin_train_root + '/ct', volume), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)  # ndarray类型，shape为(切片数, 512, 512)

        # 读取segmentation.nii文件
        seg = sitk.ReadImage(
            os.path.join(opt.origin_train_root + '/seg', volume.replace('volume', 'segmentation')), sitk.sitkInt8)
        liver_seg_array = sitk.GetArrayFromImage(seg)
        tumor_seg_array = liver_seg_array.copy()

        # 对于liver_seg_array，将1,2标签融合为1
        # 对于tumor_seg_array，将0,1标签融合为0，2标签变为1
        liver_seg_array[liver_seg_array > 0] = 1
        tumor_seg_array[tumor_seg_array < 2] = 0
        tumor_seg_array[tumor_seg_array == 2] = 1

        # seg_array和ct_array进行element-wise的乘法，提取肝脏区域
        ct_array = ct_array * liver_seg_array

        # 将灰度值在阈值之外的截断掉
        ct_array[ct_array > opt.gray_upper] = opt.gray_upper
        ct_array[ct_array < opt.gray_lower] = opt.gray_lower

        # 对切片块中的每一个切片，进行归一化操作
        ct_array = ct_array.astype(np.float32)
        ct_array = ct_array / 200.0

        # 对于肿瘤的识别，我们需要进行颜色翻转，避免肿瘤区域的颜色和背景颜色太相近，导致模型不好识别
        ct_array = 1 - ct_array  # 全部取反，此时背景区域颜色还是和肿瘤区域颜色一致
        ct_array = ct_array * liver_seg_array  # 再把背景区域颜色乘以0，变黑，最后只有肿瘤区域为亮色区域

        # 下采样原始array
        # shape变为(切片数//2,256,256)，采用双线性插值
        ct_array = ndimage.zoom(ct_array, opt.zoom_scale, order=1)
        liver_seg_array = ndimage.zoom(liver_seg_array, opt.zoom_scale, order=0)  # shape变为(切片数//2,256,256)，采用最近邻插值
        tumor_seg_array = ndimage.zoom(tumor_seg_array, opt.zoom_scale, order=0)  # shape变为(切片数//2,256,256)，采用最近邻插值

        # 搜索切片分块区间，还是找包含肝脏区域的块
        z = np.any(liver_seg_array, axis=(1, 2))  # 判断每一张切片中是否包含1（1表示肝脏），返回一个长度等于切片数的布尔数组
        start_slice, end_slice = np.where(z)[0][[0, -1]]  # np.where(z)返回数组中不为0的下标list
        start_slice = max(0, start_slice - opt.expand_slice)
        end_slice = min(liver_seg_array.shape[0], end_slice + opt.expand_slice)
        if end_slice - start_slice < opt.block_size - 1:  # 过滤掉不足以生成一个切片块的原始样本
            continue
        ct_array = ct_array[start_slice:end_slice + 1, :, :]  # 截取原始CT影像中包含肝脏区间及扩张的所有切片
        tumor_seg_array = tumor_seg_array[start_slice:end_slice + 1, :, :]
        # 开始生成厚度为48的切片块，并写入文件中，保存为nii格式
        l, r = 0, opt.block_size - 1
        while r < ct_array.shape[0]:
            # volume切片块和segmentation切片块生成
            ct_block = generate_clock(ct, ct_array, l, r)
            seg_block = generate_clock(seg, tumor_seg_array, l, r)

            ct_block_name = 'volume-' + str(idx) + '.nii'
            seg_block_name = 'segmentation-' + str(idx) + '.nii'
            sitk.WriteImage(ct_block, os.path.join(opt.train_data_root_2 + '/ct', ct_block_name))
            sitk.WriteImage(seg_block, os.path.join(opt.train_data_root_2 + '/seg', seg_block_name))

            idx += 1
            l += opt.stride
            r = l + opt.block_size - 1

        # 如果每隔opt.stride不能完整的将所有切片分块时，从后往前取到最后一个block
        if r != ct_array.shape[0] + opt.stride:
            # volume切片块生成
            ct_block = generate_clock(ct, ct_array, -opt.block_size, None)
            seg_block = generate_clock(seg, tumor_seg_array, -opt.block_size, None)

            ct_block_name = 'volume-' + str(idx) + '.nii'
            seg_block_name = 'segmentation-' + str(idx) + '.nii'
            sitk.WriteImage(ct_block, os.path.join(opt.train_data_root_2 + '/ct', ct_block_name))
            sitk.WriteImage(seg_block, os.path.join(opt.train_data_root_2 + '/seg', seg_block_name))

            idx += 1


if __name__ == '__main__':
    # 开始预处理过程
    preprocess()
