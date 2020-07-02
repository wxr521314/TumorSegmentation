# encoding: utf-8
"""
 @project:TumorSegmentation
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 12/25/19 10:30 AM
 @desc: 总共是四次下采样和四次上采样
        下采样过程，通道数增加，特征图不断减小
        上采样过程，通道数减小，特征图不断增大，每一个阶段都产生一个输出

        在创建此类的实例时，需要进行何凯明版的初始化方法

        本质上是一个逐像素点的二分类任务，所以输出层采用sigmoid激活函数
"""
from .BasicModule import BasicModule
from torch import nn
import torch.nn.functional as F
import torch as t


class CascadeResUNet(BasicModule):
    def __init__(self, training=True, net_type="liver_seg"):
        super(CascadeResUNet, self).__init__()
        self.model_name = 'cascaderesunet'
        self.training = training
        self.net_type = net_type

        # 输入切片块的shape为(1, 1, 16, 256, 256) 对应(N,C,D,H,W)
        if self.net_type == "liver_seg":
            self.encoder1 = nn.Sequential(  # 两次卷积
                nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.PReLU(num_parameters=16),
                nn.Conv3d(16, 16, 3, 1, 1),  # 3D卷积的kernel_size为(3,3,3)，对应(D,H,W)，对每个channel进行卷积
                nn.PReLU(16)
            )  # [1, 16, 16, 256, 256]
        elif self.net_type == "tumor_seg": # tumor_seg网络的输入需要算上liver_seg网络的输出部分
            
            self.shortcut = nn.Sequential( # 用于与encoder1的输出拼接，此做法类似resnet34中，输入输出大小不一致时的拼接做法
                nn.Conv3d(in_channels=32+1, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(16)
            )
            
            self.encoder1 = nn.Sequential(  # 两次卷积
                nn.Conv3d(in_channels=32+1, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.PReLU(num_parameters=16),
                nn.Conv3d(16, 16, 3, 1, 1),  # 3D卷积的kernel_size为(3,3,3)，对应(D,H,W)，对每个channel进行卷积
                nn.PReLU(16)
            )  # [1, 16, 16, 256, 256]

        self.down_sampling_1 = nn.Sequential(  # 下采样操作，通道数增加，特征图减小
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=2, stride=2),
            nn.PReLU(32)
        )  # [1, 32, 8, 128, 128]

        self.encoder2 = nn.Sequential(  # 三次卷积，通道数不变
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.PReLU(32),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.PReLU(32),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.PReLU(32)
        )  # [1, 32, 8, 128, 128]

        self.down_sampling_2 = nn.Sequential(  # 下采样操作，通道数增加，特征图减小
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=2, stride=2),
            nn.PReLU(64)
        )  # [1, 64, 4, 64, 64]

        self.encoder3 = nn.Sequential(  # 通道数不变
            nn.Conv3d(64, 64, 3, 1, 1),
            nn.PReLU(64),
            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.PReLU(64),
            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.PReLU(64)
        )  # [1, 64, 4, 64, 64]

        self.down_sampling_3 = nn.Sequential(  # 下采样操作，通道数增加，特征图减小
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=2, stride=2),
            nn.PReLU(128)
        )  # [1, 128, 2, 32, 32]

        self.encoder4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.PReLU(128),
            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.PReLU(128),
            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.PReLU(128)
        )  # [1, 128, 2, 32, 32]

        self.down_sampling_4 = nn.Sequential(  # 注意这里并没有改变特征图的大小
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(256)
        )  # [1, 256, 2, 32, 32]

        self.decoder1 = nn.Sequential(  # encoder4的输出作为decoder1的输入
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(256),
            nn.Conv3d(256, 256, 3, 1, 1),
            nn.PReLU(256),
            nn.Conv3d(256, 256, 3, 1, 1),
            nn.PReLU(256)
        )  # [1, 256, 2, 32, 32]

        self.output1 = nn.Sequential(  # decoder1的输出，shape为(N,C,D_in*8,H_in*16,W_in*16)，采用三线性插值法
            nn.Conv3d(256, 1, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear'),
            nn.Sigmoid()
        )  # [1, 1, 16, 256, 256]

        self.up_sampling_1 = nn.Sequential(  # 上采样操作，通道数减小，特征图变大
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )  # [1, 128, 4, 64, 64]

        self.decoder2 = nn.Sequential(  # 与encoder3的输出进行拼接，最后输出的通道数等于拼接前的通道数
            nn.Conv3d(128 + 64, 128, 3, 1, 1),
            nn.PReLU(128),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.PReLU(128),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.PReLU(128)
        )  # [1, 128, 4, 64, 64]

        self.output2 = nn.Sequential(  # decoder2的输出
            nn.Conv3d(128, 1, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear'),
            nn.Sigmoid()
        )  # [1, 1, 16, 256, 256]

        self.up_sampling_2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )  # [1, 64, 8, 128, 128]

        self.decoder3 = nn.Sequential(  # 与encoder2的输出进行拼接，最后输出的通道数等于拼接前的通道数
            nn.Conv3d(64 + 32, 64, 3, 1, 1),
            nn.PReLU(64),
            nn.Conv3d(64, 64, 3, 1, 1),
            nn.PReLU(64),
            nn.Conv3d(64, 64, 3, 1, 1),
            nn.PReLU(64)
        )  # [1, 64, 8, 128, 128]

        self.output3 = nn.Sequential(  # decoder3的输出
            nn.Conv3d(64, 1, 1, 1),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear'),
            nn.Sigmoid()
        )  # [1, 1, 16, 256, 256]

        self.up_sampling_3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )  # [1, 32, 16, 256, 256]

        self.decoder4 = nn.Sequential(  # 与encoder1的输出进行拼接，最后输出的通道数等于拼接前的通道数
            nn.Conv3d(32 + 16, 32, 3, 1, 1),
            nn.PReLU(32),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.PReLU(32)
        )  # [1, 32, 16, 256, 256]

        self.output4 = nn.Sequential(  # decoder2的输出
            nn.Conv3d(32, 1, 1, 1),
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear'),
            nn.Sigmoid()
        )  # [1, 1, 16, 256, 256]

    def forward(self, inputs):
        # 首先是下采样过程
        if self.net_type == 'liver_seg':
            out1 = self.encoder1(inputs) + inputs
        elif self.net_type == 'tumor_seg': # 输入输出大小不一致，于是调整输入的大小，使得能进行残差连接
            out1 = self.encoder1(inputs) + self.shortcut(inputs)
        out2 = self.down_sampling_1(out1)

        out3 = self.encoder2(out2) + out2
        out3 = F.dropout(out3, 0.3, self.training)
        out4 = self.down_sampling_2(out3)

        out5 = self.encoder3(out4) + out4
        out5 = F.dropout(out5, 0.3, self.training)
        out6 = self.down_sampling_3(out5)

        out7 = self.encoder4(out6) + out6
        out7 = F.dropout(out7, 0.3, self.training)
        out8 = self.down_sampling_4(out7)

        # 然后是上采样过程
        out9 = self.decoder1(out7) + out8
        out9 = F.dropout(out9, 0.3, self.training)
        output1 = self.output1(out9)
        out10 = self.up_sampling_1(out9)

        out11 = self.decoder2(t.cat((out10, out5), dim=1)) + out10
        out11 = F.dropout(out11, 0.3, self.training)
        output2 = self.output2(out11)
        out12 = self.up_sampling_2(out11)

        out13 = self.decoder3(t.cat((out12, out3), dim=1)) + out12
        out13 = F.dropout(out13, 0.3, self.training)
        output3 = self.output3(out13)
        out14 = self.up_sampling_3(out13)

        out15 = self.decoder4(t.cat((out14, out1), dim=1)) + out14
        out15 = F.dropout(out15, 0.3, self.training)
        output4 = self.output4(out15)
        
        if self.training:  # 训练阶段，out15是喂给下一个神经网络的数据，output是当前神经网络的输出，即分割结果
            return out15, output4
        else:
            return out15, output4
