# encoding: utf-8
"""
 @project:TumorSegmentation
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 12/27/19 6:25 PM
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
import numpy as np
import warnings
import pickle
import copy
import scipy.ndimage as ndimage
from medpy import metric

warnings.filterwarnings("ignore")
opt = DefaultConfig()

opt.liver_model_path = "checkpoints/dilated-dense-unet_0523_13:10:49.pth"


# 验证肝脏分割模型的效果
def val_liver_net():
    # 第一步：加载模型（模型，预训练参数，GPU），并设置为推理模式
    # model = getattr(models, opt.model)(training=False)  # 等价于 models.CascadeResUNet(net_type="liver_seg")
    model = models.DilatedDenseUNet()
    if opt.liver_model_path:
        print("current model path is: ", opt.liver_model_path)
        model.load(opt.liver_model_path)  # 加载模型参数
    if opt.use_gpu:
        model.cuda(opt.device)

    model.eval()

    # 第二步：从 data/val_volumes_list.txt文件中，读取验证集的文件名
    with open('data/val_volumes_list.txt', 'rb') as f:
        volumes = pickle.load(f)

    # 统计最终的平均Dice值
    total_dice = 0
    dice_intersection = 0
    dice_union = 0

    for volume_name in tqdm(volumes, total=len(volumes)):
        ct = sitk.ReadImage(os.path.join(opt.origin_train_root + '/ct', volume_name), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        # 将灰度值在阈值之外的截断掉
        ct_array[ct_array > opt.gray_upper] = opt.gray_upper
        ct_array[ct_array < opt.gray_lower] = opt.gray_lower

        # 对切片块中的每一个切片，进行归一化操作
        ct_array = ct_array.astype(np.float32)
        ct_array = ct_array / 200

        # 下采样原始array，三次插值法
        ct_array = ndimage.zoom(ct_array, opt.zoom_scale, order=1)

        # 如果原始CT影像切片数不足48，进行padding操作，将原始CT影像前面不变，后面补上若干张切片，使得总深度为48
        too_small = False
        slice_num = ct_array.shape[0]
        if slice_num < opt.block_size:
            temp = np.ones((opt.block_size, int(512 * opt.zoom_scale), int(512 * opt.zoom_scale))) * (
                    opt.gray_lower / 200.0)
            temp[0: slice_num] = ct_array
            ct_array = temp
            too_small = True

        # 将原始CT影像分割成长度为48的一系列的块，如0~47, 48~95, 96~143, .....
        start_slice, end_slice = 0, opt.block_size - 1
        count = np.zeros((ct_array.shape[0], int(512 * opt.zoom_scale), int(512 * opt.zoom_scale)),
                         dtype=np.int16)  # 用来统计原始CT影像中的每一个像素点被预测了几次
        probability_map = np.zeros((ct_array.shape[0], int(512 * opt.zoom_scale), int(512 * opt.zoom_scale)),
                                   dtype=np.float32)  # 用来存储每个像素点的预测值

        with torch.no_grad():
            while end_slice < ct_array.shape[0]:
                ct_tensor = torch.as_tensor(ct_array[start_slice: end_slice + 1]).float()
                if opt.use_gpu:
                    ct_tensor = ct_tensor.cuda(opt.device)
                ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)  # shape变为: (1, 1, 48, 256, 256)

                output = model(ct_tensor)
                count[start_slice:end_slice + 1] += 1
                probability_map[start_slice:end_slice + 1] += np.squeeze(output.cpu().detach().numpy())

                # 将输出结果转为ndarray类型，保存在CPU上，再释放掉output，减轻GPU压力
                del output
                start_slice += opt.block_size
                end_slice = start_slice + opt.block_size - 1

            # 如果原始图像的切片数超过了48，且不能被48整除，对最后一个块进行处理
            if slice_num > opt.block_size and slice_num % opt.block_size:
                end_slice = slice_num - 1
                start_slice = end_slice - opt.block_size + 1
                ct_tensor = torch.as_tensor(ct_array[start_slice: end_slice + 1]).float()
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
            liver_seg = copy.deepcopy(pred_seg)

            # 4连通只考虑4个邻接像素，即上下左右,8连通总共考虑8个邻接像素，还包括了对角线位置的点
            # 在二值图像中，如果两个像素点相邻且值相同（同为0或同为1），那么就认为这两个像素点在一个相互连通的区域内。
            # 而同一个连通区域的所有像素点，都用同一个数值来进行标记，这个过程就叫连通区域标记。在判断两个像素是否相邻时，我们通常采用4连通或8连通判断。

            measure.label(liver_seg, 4)  # 返回的是一个4连通标记，a与b连通，b与c连通，则a与c连通，根据此原则将原二值图划分为一系列的连通区域
            liver_seg = measure.label(liver_seg, 4)
            # 如果想分别上面的的每一个连通区域进行操作，比如计算面积、外接矩形、凸包面积等，则需要用到measure.regionprops,返回一个迭代器
            props = measure.regionprops(liver_seg)

            # 这里是遍历所有连通区域的面积，然后只取最大连通区域，其他小区域即过滤掉。
            max_area = 0
            max_index = 0
            for index, prop in enumerate(props, start=1):
                if prop.area > max_area:
                    max_area = prop.area
                    max_index = index

            liver_seg[liver_seg != max_index] = 0
            liver_seg[liver_seg == max_index] = 1

            # 填充最大连通区域中的小空洞区域
            liver_seg = liver_seg.astype(np.bool)  # 0变成False, 1变成True
            # remove_small_holes(ar, area_threshold=64, connectivity=1, in_place=False, min_size=None)
            # 低于area_threshold值的区域会被remove掉，即False变成True,等于的不会被删，connectivity=1表示空洞区域进行4连通，2表示8连通
            morphology.remove_small_holes(liver_seg, opt.maximum_hole, connectivity=2, in_place=True)
            liver_seg = liver_seg.astype(np.uint8)

            # 读入医生标注的segmentation.nii文件，计算评估指标
            seg = sitk.ReadImage(
                os.path.join(opt.origin_train_root + '/seg', volume_name.replace('volume', 'segmentation')),
                sitk.sitkInt8)
            seg_array = sitk.GetArrayFromImage(seg)
            seg_array[seg_array > 0] = 1
            seg_array = ndimage.zoom(seg_array, opt.zoom_scale, order=0)

            dice = metric.dc(liver_seg, seg_array)
            dice_intersection += (liver_seg * seg_array).sum() * 2
            dice_union += liver_seg.sum() + seg_array.sum()

            print("{}文件的Dice系数为{}".format(volume_name, dice))

            total_dice += dice

            # 将模型预测的文件，保存为nii格式，持久化存储，供后面肿瘤分割模型调用
            liver_seg = sitk.GetImageFromArray(liver_seg)
            liver_seg.SetDirection(seg.GetDirection())
            liver_seg.SetOrigin(seg.GetOrigin())
            liver_seg.SetSpacing(seg.GetSpacing())

            sitk.WriteImage(liver_seg, os.path.join(opt.pred_liver_root,
                                                    volume_name.replace('volume', 'pred')))  # 将预测结果写入
            del liver_seg

    print("Dice avg is {}".format(total_dice / len(volumes)))
    print("Dice global is {}".format(dice_intersection / dice_union))


if __name__ == '__main__':
    val_liver_net()
