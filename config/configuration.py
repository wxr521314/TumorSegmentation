# encoding: utf-8
"""
 @project:TumorSegmentation
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 12/25/19 9:35 AM
 @desc: 模型的配置文件
"""


class DefaultConfig(object):
    env = "LiverSegDDUNet0519"
    # model = 'ResUNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    # 路径设置
    origin_train_root = '/home/jh/LiTS2017/origin_train'  # 官网下载的原始数据
    origin_test_root = '/home/jh/LiTS2017/origin_test'  # 官网下载的原始数据
    train_data_root = '/home/jh/LiverTumorSegmentation/train'  # 随机划分的训练集文件路径（肝脏分割）
    train_data_root_2 = '/home/jh/LiverTumorSegmentation/train_tumor'  # 基于肝脏分割区间，深度维的缩放为1（肿瘤分割）
    pred_liver_root = '/home/jh/LiverTumorSegmentation/predict/liver'  # 模型预测生成的肝脏分割文件的存放路径
    pred_tumor_root = '/home/jh/LiverTumorSegmentation/predict/tumor'  # 模型预测生成的肿瘤分割文件的存放路径
    pred_tumor_root2 = '/home/jh/LiverTumorSegmentation/predict/tumor_2'  # 模型预测生成的肿瘤分割文件的存放路径

    # liver_model_path = 'checkpoints/resunet_0226_02:19:30.pth'
    # tumor_model_path = "checkpoints/resunet_0216_04:47:45.pth"
    liver_model_path = None
    tumor_model_path = None
    # 数据预处理过程
    expand_slice = 20  # 轴向 扩张的切片数量，在设置分块区间时用到
    slice_thickness = 1  # z轴分辨率
    block_size = 16  # 预处理后，块的深度
    stride = 3  # 每隔3个切片，生成一个数据块
    gray_upper = 200  # 灰度值上界截断，原始的CT影像中，最大值可到3000
    gray_lower = -200  # 灰度值下界截断，原始的CT影像中，最小值可到-3000
    zoom_scale = 0.5  # 插值算法的比例，0.5表示下采样

    # 训练过程
    device = 1
    batch_size = 2
    use_gpu = True
    num_workers = 8  # 并行加载数据
    print_freq = 10  # 每50个sample在visdom中输出一次
    max_epoch = 30
    lr = 0.0001
    lr_decay = 0.95  # 学习率的衰减率, lr *= lr_decay
    alpha = 0.3  # 加权损失
    weight_decay = 1e-4  # 损失函数

    # 验证过程
    threshold = 0.7  # 针对sigmoid的输出结果，大于等于0.7的判为1，小于0.7的判为0
    maximum_hole = 5e4  # 最大的空洞面积
