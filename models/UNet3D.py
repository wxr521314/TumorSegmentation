# encoding: utf-8
"""
 @project:TumorSegmenation
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 2020/2/14 15:56
 @desc: 3D-UNet model
"""

from .BasicModule import BasicModule
import torch
import torch.nn as nn


def conv_block_3d(in_channels, out_channels, activation):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_channels),
        activation, )


def conv_trans_block_3d(in_channels, out_channels, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_channels),
        activation, )


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


# 每个encoder内部有两次卷积操作,这里是第一个encoder，卷积操作跟后面不同，所以单独写出来
def conv_block_1_3d(in_channels, out_channels, activation):
    return nn.Sequential(
        conv_block_3d(in_channels, out_channels, activation),
        conv_block_3d(out_channels, 2 * out_channels, activation),
    )


# 这里是其他encoder的结构，卷积操作一致
def conv_block_2_3d(in_channels, out_channels, activation):
    return nn.Sequential(
        conv_block_3d(in_channels, in_channels, activation),
        conv_block_3d(in_channels, out_channels, activation),
    )


# 这里是decoder的结构，卷积操作一致
def conv_block_3_3d(in_channels, out_channels, activation):
    return nn.Sequential(
        conv_block_3d(in_channels, out_channels, activation),
        conv_block_3d(out_channels, out_channels, activation),
    )


class UNet3D(BasicModule):
    def __init__(self, in_channels=1, out_channels=1, num_filters=32):
        super(UNet3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)

        # down sampling
        self.encoder_1 = conv_block_1_3d(self.in_channels, self.num_filters, activation)
        self.down_1 = max_pooling_3d()
        self.encoder_2 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.down_2 = max_pooling_3d()
        self.encoder_3 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.down_3 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)

        # Up sampling
        self.up_1 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.decoder_1 = conv_block_3_3d(self.num_filters * (8 + 16), self.num_filters * 8, activation)
        self.up_2 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.decoder_2 = conv_block_3_3d(self.num_filters * (4 + 8), self.num_filters * 4, activation)
        self.up_3 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.decoder_3 = conv_block_3_3d(self.num_filters * (2 + 4), self.num_filters * 2, activation)

        # Output,根据论文中的代码，输出层采用的是1*1*1的卷积，考虑到这是一个像素级的二分类任务，激活函数换为Sigmoid()
        # 1*1*1的卷积就不需要padding了，3*3*3的卷积需要设定padding=(1,1,1)
        self.out = nn.Sequential(
                    nn.Conv3d(self.num_filters * 2, self.out_channels, kernel_size=1, stride=1),
                    nn.BatchNorm3d(self.out_channels),
                    nn.Sigmoid()
                   )

    def forward(self, x):
        # Down sampling
        encoder_1 = self.encoder_1(x)  # -> [1, 64, 32, 256, 256]
        down_1 = self.down_1(encoder_1)  # -> [1, 64, 16, 128, 128]

        encoder_2 = self.encoder_2(down_1)  # -> [1, 128, 16, 128, 128]
        down_2 = self.down_2(encoder_2)  # -> [1, 128, 8, 64, 64]

        encoder_3 = self.encoder_3(down_2)  # -> [1, 256, 8, 64, 64]
        down_3 = self.down_3(encoder_3)  # -> [1, 256, 4, 32, 32]

        # Bridge
        bridge = self.bridge(down_3)  # -> [1, 512, 4, 32, 32]

        # Up sampling
        up_1 = self.up_1(bridge)  # -> [1, 512, 8, 64, 64]
        concat_1 = torch.cat([up_1, encoder_3], dim=1)  # -> [1, 256+512, 8, 64, 64]
        decoder_1 = self.decoder_1(concat_1)  # -> [1, 256, 8, 64, 64]

        up_2 = self.up_2(decoder_1)  # -> [1, 256, 16, 128, 128]
        concat_2 = torch.cat([up_2, encoder_2], dim=1)  # -> [1, 256+128, 16, 128, 128]
        decoder_2 = self.decoder_2(concat_2)  # -> [1, 128, 16, 128, 128]

        up_3 = self.up_3(decoder_2)  # -> [1, 128, 32, 256, 256]
        concat_3 = torch.cat([up_3, encoder_1], dim=1)  # -> [1, 64+128, 32, 256, 256]
        decoder_3 = self.decoder_3(concat_3)  # -> [1, 64, 32, 256, 256]

        # Output
        out = self.out(decoder_3)  # -> [1, 1, 32, 256, 256]
        return out


# if __name__ == "__main__":
#     x = torch.Tensor(1, 1, 32, 256, 256).cuda() # 如果深度取48，显存会不够用
#     print("x size: {}".format(x.size()))

#     model = UNet3D().cuda()

#     out = model(x)
#     print("out size: {}".format(out.size()))
