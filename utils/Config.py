import numpy as np


class Config:
    use_gpu = True
    num_class = 19
    ignore_label = 255
    # lossfileroot = '/home/mist/Final_code/train_loss.txt'
    # data_root = '/home/mist/newnew'
    # ori_model_root = '/home/mist/Final_code/parameter/cityscapes-bisenet-R18.pth'
    # fast_model_root = '/home/mist/Final_code/parameter/cityscapes-bisenet-X39.pth'
    lossfileroot = '/home/froven/桌面/Final_code/train_loss.txt'
    data_root = '/home/froven/桌面/newdata'
    ori_model_root = '/home/froven/桌面/Final_code/parameter/cityscapes-bisenet-R18.pth'
    fast_model_root = '/home/froven/桌面/Final_code/parameter/cityscapes-bisenet-X39.pth'
    colormap = [[128, 64, 128], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
                [107, 142, 35], [0, 200, 50], [152, 251, 152], [70, 130, 180], [0, 180, 180],
                [255, 0, 0], [0, 0, 142], [200, 100, 0], [0, 0, 0]]

    cm2lbl = np.zeros(256 ** 3)  # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
    for i, cm in enumerate(colormap):
        cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i  # 建立索引

    classname = ['road', 'building', 'fence', 'pole', 'rock', 'tree', 'bush',
                 'terrain', 'sky', 'water', 'person', 'car', 'ship', 'ignore']

    lb2image = np.array(colormap).astype('uint8')

    image_mean = np.array([0.485, 0.456, 0.406])  # 0.485, 0.456, 0.406
    image_std = np.array([0.229, 0.224, 0.225])

    epoch = 2
    step = 10
    # the parameters which need to change
    is_fast = False
    train_batch_size = 2
    crop_size_h = 256
    crop_size_w = 256

    test_batch_size = 1
    transfer_batch_size = train_batch_size
    epoch_size = 100
    num_workers = 1
    source = np.zeros((32, 32), dtype=np.int64)
    target = np.ones((32, 32), dtype=np.int64)


config = Config()
