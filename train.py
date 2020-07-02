# encoding: utf-8
"""
 @project:TumorSegmentation
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 12/25/19 9:35 AM
 @desc: train_liver_net() 负责肝脏分割网络模型的训练
        train_tumor_net() 负责肿瘤分割网络模型的训练
        每完成一个epoch就保存一个模型到checkpoints文件夹中，并在之前设计的验证集上评估，val_set[:16]为验证集，val_set[16:]为测试集
        验证集上的指标为dice avg和dice global，在新窗口可视化展示
"""
from skimage import measure
from skimage import morphology
import os
import SimpleITK as sitk
from medpy import metric
import numpy as np
import copy
import scipy.ndimage as ndimage

import torch
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm
from data.dataset2 import CascadeData
from data.dataset import Liver, Tumor
from loss.DiceLoss import DiceLoss
from loss.TverskyLoss import TverskyLoss
from loss.LovaszLoss import lovasz_hinge
from utils.visualize import Visualizer
import models
import pickle
from config.configuration import DefaultConfig
from torch import nn
import warnings

warnings.filterwarnings("ignore")
opt = DefaultConfig()
vis = Visualizer(opt.env)


def initial_params(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(module.weight.data, 0.25)
        nn.init.constant(module.bias.data, 0)


def train_liver_net():
    # 第一步：加载模型（模型，预训练参数，GPU）
    #     model = getattr(models, opt.model)(net_type="liver_seg")  # 等价于 models.ResUNet()
    model = models.DilatedDenseUNet()  # 等价于 models.ResUNet()
    # model = models.ResUNet()  # 等价于 models.ResUNet()
    if opt.liver_model_path:
        print("current model path is: ", opt.liver_model_path)
        model.load(opt.liver_model_path)  # 加载模型参数
    else:
        model.apply(initial_params)  # 网络参数初始化
        print('default initail_params')
    if opt.use_gpu:
        model.cuda(opt.device)

    # 第二步：加载数据（训练集，用DataLoader来装载）
    train_data = Liver()
    train_data_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    # 第三步：定义损失函数和优化器
    criterion = DiceLoss()
    if opt.use_gpu:
        criterion = criterion.cuda(opt.device)
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_decay)

    # 第四步：定义评估指标，这里用训练集上的平均损失
    loss_meter = meter.AverageValueMeter()  # loss_meter.value()，返回一个二元组，第一个元素是均值，第二个元素是标准差

    # 第五步：开始训练过程
    for epoch in range(opt.max_epoch):
        loss_meter.reset()  # 置为(nan,nan)
        import math
        for ii, (liver, seg) in tqdm(enumerate(train_data_loader), total=math.ceil(len(train_data) / opt.batch_size)):
            if opt.use_gpu:
                liver = liver.cuda(opt.device)
                seg = seg.cuda(opt.device)
            optimizer.zero_grad()  # 每轮都要清空一轮梯度信息
            output = model(liver)
            seg[seg > 1] = 1
            # 计算损失，反向传播
            loss = criterion(output, seg)  # 原来是计算每一个encoder层的损失，我们这里只处理最后一个encoder的损失
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新网络权重参数
            loss_meter.add(loss.item())  # 将当前batch的损失加入到统计指标中

            if ii % opt.print_freq == opt.print_freq - 1:  # 每20个batch可视化一次损失
                vis.plot('Dice Loss', loss_meter.value()[0])
                vis.log(
                    'lr:{lr},loss:{loss}'.format(lr=optimizer.state_dict()['param_groups'][0]['lr'],
                                                 loss=loss_meter.value()[0]))

        # 每完成一个epoch的训练，就保存一轮模型,并衰减一下学习率
        model.save()  # 存在checkpoints目录下
        lr_scheduler.step()


def train_tumor_net():
    # 第一步：加载模型（模型，预训练参数，GPU）
    model = models.ResUNet()
    if opt.tumor_model_path:
        print("current model path is: ", opt.tumor_model_path)
        model.load(opt.tumor_model_path)  # 加载模型参数
    else:
        print('initial_params')
        model.apply(initial_params)  # 网络参数初始化
    if opt.use_gpu:
        model.cuda(opt.device)

    # 第二步：加载数据（训练集，用DataLoader来装载）
    train_data = Tumor()
    train_data_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    # 第三步：定义损失函数和优化器，学习率衰减
    criterion = DiceLoss()
    if opt.use_gpu:
        criterion = criterion.cuda(opt.device)
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_decay)

    # 第四步：定义评估指标，这里用训练集上的平均损失
    loss_meter = meter.AverageValueMeter()  # loss_meter.value()，返回一个二元组，第一个元素是均值，第二个元素是标准差

    # 第五步：开始训练过程
    for epoch in range(opt.max_epoch):
        loss_meter.reset()  # 置为(nan,nan)
        import math
        for ii, (tumor, seg) in tqdm(enumerate(train_data_loader), total=math.ceil(len(train_data) / opt.batch_size)):
            if opt.use_gpu:
                tumor = tumor.cuda(opt.device)
                seg = seg.cuda(opt.device)
            optimizer.zero_grad()  # 每轮都要清空一轮梯度信息
            output = model(tumor)

            # 计算损失，反向传播
            loss = criterion(output, seg)  # 原来是计算每一个encoder层的损失，我们这里只处理最后一个encoder的损失
            #             loss = criterion(outputs, seg)
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新网络权重参数
            loss_meter.add(loss.item())  # 将当前batch的损失加入到统计指标中

            if ii % opt.print_freq == opt.print_freq - 1:  # 每20个batch可视化一次损失
                vis.plot('Dice Loss', loss_meter.value()[0])
                vis.log(
                    'lr:{lr},loss:{loss}'.format(lr=optimizer.state_dict()['param_groups'][0]['lr'],
                                                 loss=loss_meter.value()[0]))

        # 每完成一个epoch的训练，就保存一轮模型,并衰减一下学习率
        model.save()  # 存在checkpoints目录下
        #         val_tumor_net(model)  # 计算验证误差并可视化
        lr_scheduler.step()


def val_tumor_net(model):
    # 模型设置为评估模式
    model.eval()
    model.training = False

    with open('data/val_volumes_list.txt', 'rb') as f:
        volumes = pickle.load(f)[:16]

    # 统计Dice avg 和 Dice global
    total_dice, dice_intersection, dice_union = 0, 0, 0

    for volume in tqdm(volumes, total=len(volumes)):
        # 读取volume.nii文件
        ct = sitk.ReadImage(os.path.join(opt.origin_train_root + '/ct', volume), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)  # ndarray类型，shape为(切片数, 512, 512)

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

        # 下采样原始array，三次插值法
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

            pred_seg = pred_seg.astype(np.uint8)
            tumor_seg = copy.deepcopy(pred_seg)

            # 读入医生标注的segmentation.nii文件，计算评估指标
            seg = sitk.ReadImage(
                os.path.join(opt.origin_train_root + '/seg', volume.replace('volume', 'segmentation')),
                sitk.sitkInt8)
            seg_array = sitk.GetArrayFromImage(seg)

            seg_array[seg_array < 2] = 0
            seg_array[seg_array == 2] = 1
            seg_array = ndimage.zoom(seg_array, opt.zoom_scale, order=0)  # label采取最近邻插值法

            dice = metric.dc(tumor_seg, seg_array)
            dice_intersection += (tumor_seg * seg_array).sum() * 2
            dice_union += tumor_seg.sum() + seg_array.sum()
            total_dice += dice

            del tumor_seg

    # 验证集上的指标为dice avg和dice global，在新窗口可视化展示
    vis.plot_two_line('Dice Coefficient', total_dice / len(volumes), dice_intersection / dice_union,
                      legend_name=['dice avg', 'dice global'])

    # 模型恢复为训练模式
    model.train()
    model.training = True


if __name__ == '__main__':
    import fire

    fire.Fire()
