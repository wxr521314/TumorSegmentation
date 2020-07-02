# encoding: utf-8
"""
 @project:TumorSegmentation
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 12/25/19 9:36 AM
 @desc:
"""
import torch
import torch.nn as nn
import time
from config.configuration import DefaultConfig
opt = DefaultConfig()


class BasicModule(nn.Module):
    """
    封装了nn.Module，主要提供了save和load两个方法
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))  # 模型的默认名字

    def load(self, path):
        """
        :param path: 模型的路径
        :return: 返回指定路径的模型
        """
#         self.load_state_dict(torch.load(path))
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage.cuda(opt.device)))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        如AlexNet_1210_20:20:29.pth
        """
        if not name:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime((prefix + '%m%d_%H:%M:%S.pth'))
        torch.save(self.state_dict(), name)
        return name
