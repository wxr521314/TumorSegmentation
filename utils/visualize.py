# encoding: utf-8
"""
 @project:TumorSegmentation
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 12/25/19 9:37 AM
 @desc:
"""
import visdom
import time
import numpy as np


class Visualizer(object):
    """
    封装了visdom的基本操作，但是你仍可以通过"self.vis.function"或"self.function"方式调用原生的visdom接口
    比如：
    self.text("hello visdom")
    self.histogram(torch.randn(1000))
    self.line(torch.arrange(0,10),torch.arrange(1,11))
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        self.index = {}  # 是一个字典，{'loss':23}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个点
        @params d: dict{name, value} i.e. {'loss' : 0.11}
        """
        for k, v in d.iteritems():
            self.plot(k, v)

    def img_many(self, d):
        """
        一次绘制多张图
        """
        for k, v in d.iteritems():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        调用方法：self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)  # 字典对象，如果存在name这个key，就返回相应的值，不存在该key，就返回0
        self.vis.line(Y=np.array([y]), X=np.array([x]), win=name, opts=dict(title=name),
                      update=None if x == 0 else 'append', **kwargs)
        self.index[name] = x + 1
        
    def plot_two_line(self, name, y1, y2, **kwargs):
        """
        调用方法：self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)  # 字典对象，如果存在name这个key，就返回相应的值，不存在该key，就返回0
        self.vis.line(Y=np.column_stack((np.array([y1]),np.array([y2]))), X=np.array([x]), win=name, opts=dict(title=name,
        legend=['Liver seg', 'Tumor seg']),
                      update=None if x == 0 else 'append', **kwargs)
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img', t.Tensor(64, 64))
        self.img('input_imgs', t.Tensor(3, 64, 64))
        self.img('input_imgs', t.Tensor(100, 1, 64, 64))
        self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
        """
        self.vis.images(img_.cpu().numpy(), win=name, opts=dict(title=name), **kwargs)

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1, 'lr':0.0001})
        """
        self.log_text += ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        """
        self.function 等价于self.vis.function
        自定义的plot,image,log,plot_many等除外
        """
        return getattr(self.vis, name)
