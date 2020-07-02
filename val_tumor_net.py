# encoding: utf-8
"""
 @project:TumorSegmentation
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 1/5/20 10:46 AM
 @desc: 用训练好的模型对验证集文件进行测试，主要流程如下：
        (1) 读取验证集文件中的原始CT影像和GroundTruth
        (2) 对原始CT影像进行预处理（截断、归一化、下采样、分块） 重点聊一下分块操作
            (2.1) 如果原始影像的切片数小于opt.block_size(即48)，使用padding技术
            (2.2) start_slice=0，每隔47张切片截取一段，即下一个块的start_slice=48，以此类推
            (2.3) 如果原始影像的切片数可以刚好被48整除，则不做后续处理，否则：
                  从原始影像的最后一张切片开始，向前取47张切片，作为一个块进行预测。
                  最后合并时，只取 output[-切片总数%48:] 张切片

        (3) 沿轴向合并模型生成的所有深度为48的块，并对模型分割结果进行后处理（移除细小区域,并进行内部的空洞填充）
        (4) 评估模型分割指标(Dice系数)，并将预测结果写入到文件中

"""
from skimage import measure
from skimage import morphology
import torch
import os
from tqdm import tqdm
import models
from config.configuration import DefaultConfig
import SimpleITK as sitk
from medpy import metric
import numpy as np
import warnings
import pickle
import copy
import scipy.ndimage as ndimage

warnings.filterwarnings("ignore")
opt = DefaultConfig()

opt.tumor_model_path = "checkpoints/resunet_0216_04:47:45.pth"

# 验证肝脏分割模型的效果
def val_tumor_net():
    # 第一步：加载模型（模型，预训练参数，GPU），并设置为推理模式
    model = models.ResUNet()
    if opt.tumor_model_path:
        print("current model path is: ", opt.tumor_model_path)
        model.load(opt.tumor_model_path)  # 加载模型参数
    if opt.use_gpu:
        model.cuda(opt.device)

    model.eval()

    # 第二步：从 data/val_volumes_list.txt文件中，读取验证集的文件名
    with open('data/val_volumes_list.txt', 'rb') as f:
        volumes = pickle.load(f)

    # 统计Dice avg 和 Dice global
    total_dice = 0
    dice_intersection = 0
    dice_union = 0

    for volume in tqdm(volumes, total=len(volumes)):
        # 读取volume.nii文件
        ct = sitk.ReadImage(os.path.join(opt.origin_train_root + '/ct', volume), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)  # ndarray类型，shape为(切片数, 512, 512)
        # 下采样原始array，三次插值法，因为肝脏预测的array大小为256*256
        # ct_array = ndimage.zoom(ct_array, opt.zoom_scale, order=3)

        # 读取模型预测的肝脏分割文件 pred.nii
        # liver_seg = sitk.ReadImage(
        #     os.path.join(opt.pred_liver_root, volume.replace('volume', 'pred')), sitk.sitkInt8)
        # liver_seg_array = sitk.GetArrayFromImage(liver_seg)
        # liver_seg_array[liver_seg_array > 0] = 1

        # 基于GT分割肝脏
        liver_seg = sitk.ReadImage(
            os.path.join(opt.origin_train_root + '/seg', volume.replace('volume', 'segmentation')), sitk.sitkInt8)
        liver_seg_array = sitk.GetArrayFromImage(liver_seg)
        liver_seg_array[liver_seg_array > 0] = 1

        # liver_seg_array和ct_array进行element-wise的乘法，提取肝脏区域
        liver_array = liver_seg_array * ct_array

        # 将灰度值在阈值之外的截断掉
        liver_array[liver_array > opt.gray_upper] = opt.gray_upper
        liver_array[liver_array < opt.gray_lower] = opt.gray_lower

        # 对切片块中的每一个切片，进行归一化操作
        liver_array = liver_array.astype(np.float32)
        liver_array = liver_array / 200

        # 对于肿瘤的识别，我们需要进行颜色翻转，避免肿瘤区域的颜色和背景颜色太相近，导致模型不好识别
        liver_array = 1 - liver_array  # 全部取反，此时背景区域颜色还是和肿瘤区域颜色一致
        liver_array = liver_array * liver_seg_array  # 再把背景区域颜色乘以0，变黑，最后只有肿瘤区域为亮色区域

        liver_array = ndimage.zoom(liver_array, opt.zoom_scale, order=3)

        # 如果原始CT影像切片数不足48，进行padding操作，将原始CT影像前面不变，后面补上若干张切片，使得总深度为48
        too_small = False
        slice_num = liver_array.shape[0]
        if slice_num < opt.block_size:
            temp = np.ones((opt.block_size, int(512 * opt.zoom_scale), int(512 * opt.zoom_scale))) * (
                        opt.gray_lower / 200.0)
            temp[0: slice_num] = liver_array
            liver_array = temp
            too_small = True

        # 将原始CT影像分割成长度为48的一系列的块，如0~47, 48~95, 96~143, .....
        start_slice, end_slice = 0, opt.block_size - 1
        count = np.zeros((liver_array.shape[0], int(512 * opt.zoom_scale), int(512 * opt.zoom_scale)),
                         dtype=np.int16)  # 用来统计原始CT影像中的每一个像素点被预测了几次
        probability_map = np.zeros((liver_array.shape[0], int(512 * opt.zoom_scale), int(512 * opt.zoom_scale)),
                                   dtype=np.float32)  # 用来存储每个像素点的预测值

        with torch.no_grad():
            while end_slice < liver_array.shape[0]:
                ct_tensor = torch.as_tensor(liver_array[start_slice: end_slice + 1]).float()
                if opt.use_gpu:
                    ct_tensor = ct_tensor.cuda(opt.device)
                ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)  # shape变为: (1, 1, 16, 256, 256)
                output = model(ct_tensor)
                count[start_slice:end_slice + 1] += 1
                probability_map[start_slice:end_slice + 1] += np.squeeze(output.cpu().detach().numpy())

                # 将输出结果转为ndarray类型，保存在CPU上，再释放掉output，减轻GPU压力
                del output

                # 设置新的块区间
                start_slice += opt.block_size
                end_slice = start_slice + opt.block_size - 1

            # 如果原始图像的切片数超过了48，且不能被48整除，对最后一个块进行处理
            if slice_num > opt.block_size and slice_num % opt.block_size:
                end_slice = slice_num - 1
                start_slice = end_slice - opt.block_size + 1
                ct_tensor = torch.as_tensor(liver_array[start_slice: end_slice + 1]).float()
                if opt.use_gpu:
                    ct_tensor = ct_tensor.cuda(opt.device)
                ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)  # shape变为: (1, 1, 48, 256, 256)

                output = model(ct_tensor)
                count[start_slice:end_slice + 1] += 1
                probability_map[start_slice:end_slice + 1] += np.squeeze(output.cpu().detach().numpy())

                # 将输出结果转为ndarray类型，保存在CPU上，再释放掉output，减轻GPU压力
                del output

            # 针对sigmoid的输出结果，每个像素点的预测值，大于等于opt.threshold(即0.7)的判为1，小于opt.threshold的判为0
            pred_seg = np.zeros_like(probability_map)  # 生成一个shape和probability_map相同的全为零的矩阵
            pred_seg[probability_map >= opt.threshold * count] = 1  # 有的像素点预测了两次，体现了count的意义

            # 前面对于切片数量不足以48的CT影像，进行了填充操作，这里需去掉填充的假切片
            if too_small:
                pred_seg = pred_seg[:slice_num]

            # pred_seg后处理，进行最大连通域提取,移除细小区域,并进行内部的空洞填充
            pred_seg = pred_seg.astype(np.uint8)
            tumor_seg = copy.deepcopy(pred_seg)

            # 读入医生标注的segmentation.nii文件，计算评估指标
            seg = sitk.ReadImage(
                os.path.join(opt.origin_train_root + '/seg', volume.replace('volume', 'segmentation')),
                sitk.sitkInt8)
            seg_array = sitk.GetArrayFromImage(seg)
            seg_array[seg_array < 2] = 0
            seg_array[seg_array == 2] = 1
            seg_array = ndimage.zoom(seg_array, opt.zoom_scale, order=0)
            dice = metric.dc(tumor_seg, seg_array)
            dice_intersection += (tumor_seg * seg_array).sum() * 2
            dice_union += tumor_seg.sum() + seg_array.sum()
            print("{}文件的Dice系数为{}".format(volume, dice))

            total_dice += dice

            # 将模型预测的文件，保存为nii格式，持久化存储
            tumor_seg = sitk.GetImageFromArray(tumor_seg)
            tumor_seg.SetDirection(ct.GetDirection())
            tumor_seg.SetOrigin(ct.GetOrigin())
            tumor_seg.SetSpacing(ct.GetSpacing())

            sitk.WriteImage(tumor_seg, os.path.join(opt.pred_tumor_root,
                                                    volume.replace('volume', 'pred')))  # 将预测结果写入
            del tumor_seg

    print("Dice avg is {}".format(total_dice / len(volumes)))
    print("Dice global is {}".format(dice_intersection / dice_union))


if __name__ == '__main__':
    val_tumor_net()
