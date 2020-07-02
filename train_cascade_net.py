# encoding: utf-8
"""
 @project:TumorSegmenation
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 2020/2/18 20:46
 @desc: 肝脏分割和肿瘤分割网络联合训练
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm
from data.dataset2 import CascadeData
from loss.DiceLoss import DiceLoss
# from loss.TverskyLoss import TverskyLoss
# from loss.LovaszLoss import lovasz_hinge
from utils.visualize import Visualizer
import models
from config.configuration import DefaultConfig
from torch import nn
import warnings
import time

warnings.filterwarnings("ignore")
opt = DefaultConfig()


def initial_params(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(module.weight.data, 0.25)
        nn.init.constant(module.bias.data, 0)


def train_cascade_net():
    vis = Visualizer(opt.env)

    # 第一步：加载模型（模型，预训练参数，GPU）
    model_1 = models.ResUNet()  # 等价于 models.CascadeResUNet()
    model_2 = models.ResUNet()  # 等价于 models.CascadeResUNet()

    # model_1.load("checkpoints/cascaderesunet_0221_06:34:37.pth")  # 0.91661的肝脏分割模型
    # model_2.load("checkpoints/resunet_0216_04:47:45.pth")  # 0.44287的肿瘤分割模型
    model_1.apply(initial_params)
    model_2.apply(initial_params)
    if opt.use_gpu:
        model_1.cuda(opt.device)
        model_2.cuda(opt.device)

    # 第二步：加载数据（训练集，用DataLoader来装载）
    train_data = CascadeData()
    train_data_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    # 第三步：定义损失函数和优化器
    criterion = DiceLoss()
    if opt.use_gpu:
        criterion = criterion.cuda(opt.device)
    lr = opt.lr
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=lr, weight_decay=opt.weight_decay)
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=lr, weight_decay=opt.weight_decay)
    lr_scheduler_1 = torch.optim.lr_scheduler.ExponentialLR(optimizer_1, opt.lr_decay)
    lr_scheduler_2 = torch.optim.lr_scheduler.ExponentialLR(optimizer_2, opt.lr_decay)

    # 第四步：定义评估指标，这里用训练集上的平均损失
    loss_meter_1 = meter.AverageValueMeter()  # loss_meter.value()，返回一个二元组，第一个元素是均值，第二个元素是标准差
    loss_meter_2 = meter.AverageValueMeter()  # loss_meter.value()，返回一个二元组，第一个元素是均值，第二个元素是标准差

    # 第五步：开始训练过程
    for epoch in range(opt.max_epoch):
        loss_meter_1.reset()  # 置为(nan,nan)
        loss_meter_2.reset()  # 置为(nan,nan)
        import math
        for ii, (ct_array, seg_array) in tqdm(enumerate(train_data_loader),
                                              total=math.ceil(len(train_data) / opt.batch_size)):
            if opt.use_gpu:
                ct_array = ct_array.cuda(opt.device)
                seg_array = seg_array.cuda(opt.device)
            optimizer_1.zero_grad()  # 每轮都要清空一轮梯度信息
            optimizer_2.zero_grad()

            output_1 = model_1(ct_array)
            output_2 = model_2(torch.cat(ct_array, output_1), axis=1)  # 指定通道维拼接

            # 计算损失，反向传播
            tumor_seg = seg_array.clone()
            seg_array[seg_array > 1] = 1  # 肝脏分割的ground truth

            tumor_seg[tumor_seg < 2] = 0  # 肿瘤分割的ground truth
            tumor_seg[tumor_seg == 2] = 1

            loss_1 = criterion(output_1, seg_array)  # liver seg 的 loss
            loss_2 = criterion(output_2, tumor_seg)  # 原来是计算每一个encoder层的损失，我们这里只处理最后一个encoder的损失

            loss_1.backward(retain_graph=True)  # 一次forward()，多个不同loss的backward()来累积同一个网络的grad
            loss_2.backward()  # 计算梯度

            optimizer_1.step()  # 只更新第一个网络的参数
            optimizer_2.step()  # 只更新第二个网络的参数

            loss_meter_1.add(loss_1.item())  # 将当前batch的损失加入到统计指标中
            loss_meter_2.add(loss_2.item())  # 将当前batch的损失加入到统计指标中

            if ii % opt.print_freq == opt.print_freq - 1:  # 每20个batch可视化一次损失
                vis.plot_two_line('Dice Loss', loss_meter_1.value()[0], loss_meter_2.value()[0])
                vis.log(
                    'lr:{lr},liver_loss:{loss1},tumor_loss:{loss2}'.format(
                        lr=optimizer_1.state_dict()['param_groups'][0]['lr'],
                        loss1=loss_meter_1.value()[0],
                        loss2=loss_meter_2.value()[0]))

        # 每完成一个epoch的训练，就保存一轮模型,并衰减一下学习率
        #         if epoch and epoch % 2 == 0:
        prefix_1 = 'checkpoints/liver_seg_'
        prefix_2 = 'checkpoints/tumor_seg_'
        name_1 = time.strftime((prefix_1 + '%m%d_%H:%M:%S.pth'))
        name_2 = time.strftime((prefix_2 + '%m%d_%H:%M:%S.pth'))
        model_1.save(name_1)  # 存在checkpoints目录下
        model_2.save(name_2)
        lr_scheduler_1.step()
        lr_scheduler_2.step()


if __name__ == '__main__':
    train_cascade_net()
