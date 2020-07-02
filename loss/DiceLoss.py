# encoding: utf-8
"""
 @project:TumorSegmentation
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 12/25/19 9:39 AM
 @desc: 图像分割场景中常用的损失函数,DiceLoss
"""


import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        
        # 输入的pred的shape为[1,1,48,512,512]，而target的shape为[1,48,512,512]
        pred = pred.squeeze(dim=1)

        smooth = 1

        # dice系数的定义
        dice = 2 * (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (pred.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                            target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        # 返回的是dice距离,torch.clamp(input, min, max, out=None),讲张量截断在[0,1]区间里 
        return torch.clamp((1 - dice).mean(), 0, 1)

    
    