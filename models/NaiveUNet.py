# encoding: utf-8
"""
 @project:TumorSegmentation
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 5/15/20 9:48 AM
 @desc:
"""
from .BasicModule import BasicModule
from torch import nn
import torch.nn.functional as F
import torch as t


class NaiveUNet(BasicModule):
    def __init__(self):
        super(NaiveUNet, self).__init__()
        self.model_name = 'naive-unet'

        self.stage1_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=16)
        )
        self.stage1_conv2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=16)
        )
        self.down1 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=2, stride=2),
            nn.PReLU(num_parameters=32)
        )
        self.stage2_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=32)
        )
        self.stage2_conv2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=32)
        )
        self.stage2_conv3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=32)
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=2, stride=2),
            nn.PReLU(num_parameters=64)
        )
        self.stage3_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.PReLU(num_parameters=64)
        )
        self.stage3_conv2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=64)
        )
        self.stage3_conv3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=64)
        )
        self.down3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=2, stride=2),
            nn.PReLU(num_parameters=128)
        )
        self.stage4_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=128)
        )
        self.stage4_conv2 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=128)
        )
        self.stage4_conv3 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=128)
        )
        self.down4 = nn.Sequential(  # 注意这里并没有改变特征图的大小
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=256)
        )
        self.stage5_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=256)
        )
        self.stage5_conv2 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=256)
        )
        self.stage5_conv3 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=256)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.PReLU(num_parameters=128)
        )
        self.stage6_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=64 + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=128)
        )
        self.stage6_conv2 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=128)
        )
        self.stage6_conv3 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=128)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.PReLU(num_parameters=64)
        )
        self.stage7_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=32 + 64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=64)
        )
        self.stage7_conv2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=64)
        )
        self.stage7_conv3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=64)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.PReLU(num_parameters=32)
        )
        self.stage8_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=16 + 32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=32)
        )
        self.stage8_conv2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=32)
        )
        self.stage8_conv3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        s1c1 = self.stage1_conv1(inputs)
        s1c2 = self.stage1_conv2(s1c1)
        down1 = self.down1(s1c2)

        s2c1 = self.stage2_conv1(down1)
        s2c2 = self.stage2_conv2(s2c1)
        s2c3 = self.stage2_conv3(s2c2)
        down2 = self.down2(s2c3)

        s3c1 = self.stage3_conv1(down2)
        s3c2 = self.stage3_conv2(s3c1)
        s3c3 = self.stage3_conv3(s3c2)
        down3 = self.down3(s3c3)

        s4c1 = self.stage4_conv1(down3)
        s4c2 = self.stage4_conv2(s4c1)
        s4c3 = self.stage4_conv3(s4c2)
        down4 = self.down4(s4c3)

        s5c1 = self.stage5_conv1(down4)
        s5c2 = self.stage5_conv2(s5c1)
        s5c3 = self.stage5_conv3(s5c2)
        up1 = self.up1(s5c3)

        s6c1 = self.stage6_conv1(t.cat([s3c3, up1], 1))
        s6c2 = self.stage6_conv2(s6c1)
        s6c3 = self.stage6_conv3(s6c2)
        up2 = self.up2(s6c3)

        s7c1 = self.stage7_conv1(t.cat([s2c3, up2], 1))
        s7c2 = self.stage7_conv2(s7c1)
        s7c3 = self.stage7_conv3(s7c2)
        up3 = self.up3(s7c3)

        s8c1 = self.stage8_conv1(t.cat([s1c2, up3], 1))
        s8c2 = self.stage8_conv2(s8c1)
        s8c3 = self.stage8_conv3(s8c2)

        return s8c3
