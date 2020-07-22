"""
English:
    This part is FCN in semantic segmentation network, which provides three basic models: FCN-32s, FCN16s, FCN8s.
    You can use instantiate the class directly:
    'model = FCN(num_class=3, kind=8)'
    where kind means the different kind of FCN.

Chinese:
    这部分是语义分割中的ＦＣＮ模型，提供了FCN-32s, FCN16s, FCN8s三种结构,
    你可以直接实例化类:
    model = FCN(num_class=3, kind=8)
    其中kind指的是不同种类的FCN模型
"""

import numpy as np
import torch
from torch import nn

from model.basemodel.loading_model import vgg_model_load


def get_upsampling_weight(in_channels, out_channels, kernel_size):  #
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    bilinear_filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)
    weight[range(in_channels), range(out_channels), :, :] = bilinear_filter
    return torch.from_numpy(weight).float()


def bilinear_upsampling(in_channels, out_channels, kernel_size, stride, bias=False):
    # Using the parameter of bilinear to initialize the parameter of deconvolution
    # 使用双线性插值法初始化反卷积层的参数
    initial_weight = get_upsampling_weight(in_channels, out_channels, kernel_size)
    layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=bias)
    layer.weight.data.copy_(initial_weight)
    return layer


class FCN(nn.Module):
    def __init__(self, num_class, kinds=32):
        super(FCN, self).__init__()
        self.kinds = kinds
        self.num_class = num_class

        # Using the changed vgg16 to adapt any size of input.
        # It can also use the pretrained model like self.vgg16_features = vgg_model_load(16).features
        # 使用修改的vgg16模型以适应任意大小的输入，也可以直接使用预训练模型：self.vgg16_features = vgg_model_load(16).features
        self.vgg16_features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32
        )
        # Change the fully connect layer to the fully conv.
        # 将全连接层转换为全卷积的形式
        self.fc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.fc_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        # To classify the result, and get the features of the number of categories
        # 对结果进行分类,得到相应类别数量的特征图.
        self.classify_8  = nn.Conv2d(in_channels=256, out_channels=self.num_class, kernel_size=1)
        self.classify_16 = nn.Conv2d(in_channels=512, out_channels=self.num_class, kernel_size=1)
        self.classify_32 = nn.Conv2d(in_channels=4096, out_channels=self.num_class, kernel_size=1)

        # using different upsampling structures according to different kinds, which you can see the paper for details,
        # where, 'kernel_size' and 'stride' are determined by the size of the input and output.
        # 根据不同的模型使用不同的上采样结构,详见论文跳跃结构,其中kernel_size和stride由特征图大小和上采样结果的大小计算而得
        self.upsampling_32_out = bilinear_upsampling(self.num_class, self.num_class, kernel_size=64, stride=32, bias=False)

        self.upsampling_16_up = bilinear_upsampling(self.num_class, self.num_class, kernel_size=4, stride=2, bias=False)
        self.upsampling_16_out = bilinear_upsampling(self.num_class, self.num_class, kernel_size=32, stride=16, bias=False)

        self.upsampling_8_up = bilinear_upsampling(self.num_class, self.num_class, kernel_size=4, stride=2, bias=False)
        self.upsampling_8_out = bilinear_upsampling(self.num_class, self.num_class, kernel_size=32, stride=16, bias=False)

    def forward(self, x):
        _, c, h, w = x.size()
        x = self.vgg16_features[0:17](x)
        y8 = self.classify_8(x)
        x = self.vgg16_features[17:24](x)
        y16 = self.classify_16(x)
        x = self.vgg16_features[24:](x)
        x = self.fc_conv1(x)
        x = self.fc_conv2(x)
        y32 = self.classify_32(x)

        if self.kinds == 32:
            out = self.upsampling_32_out(y32)
            return out[:, :, 19:19 + h, 19:19 + w].contiguous()
        if self.kinds == 16:
            up_from_y32 = self.upsampling_16_up(y32)
            y16 = y16[:, :, 5:5 + up_from_y32.size()[2], 5:5 + up_from_y32.size()[3]] + up_from_y32
            out = self.upsampling_16_out(y16)
            return out[:, :, 27:27 + h, 27:27 + w].contiguous()
        # using the kinds of 16 or 8 needs to fuse the features by adding
        # where [:, :, 5:5 + up_from_y32.size()[2], 5:5 + up_from_y32.size()[3]]
        # means cropping the original features to make sure the size of features is the same as the other features.
        # (the same below)
        # 使用16或8的结构,需要对特征图进行融合，这里使用的是相加的方式.
        # 其中,使用[:, :, 5:5 + up_from_y32.size()[2], 5:5 + up_from_y32.size()[3]]是对原特征图进行裁剪
        # 为了保证融合的特征图大小相同 (下同)
        if self.kinds == 8:
            up_from_y32 = self.upsampling_16_up(y32)
            y16 = y16[:, :, 5:5 + up_from_y32.size()[2], 5:5 + up_from_y32.size()[3]] + up_from_y32
            up_from_y16 = self.upsampling_8_up(y16)
            y8 = y8[:, :, 9:9 + up_from_y16.size()[2], 9:9 + up_from_y16.size()[3]] + up_from_y16
            out = self.upsampling_8_out(y8)
            return out[:, :, 31:31 + h, 31:31 + w].contiguous()


if __name__ == "__main__":
    # you can run this part directly to check the structure of model and the output of model
    # 你可以直接运行该部分,查看模型的结构和最后输出的特征图的大小
    model = FCN(num_class=19, kinds=8)
    # print(model)
    # _in = torch.rand((3, 512, 512)).unsqueeze(0).cuda()
    # _out = model(_in)
    # print(_out.size())
    #### test model
    from torchstat import stat
    import time
    # stat(model, (3, 512, 512))
    data = torch.rand((1, 3, 512, 512)).cuda()
    model = model.cuda()
    model.eval()
    epoch_start_time = time.time()
    predict = model(data)
    predict = predict.max(1)[1].squeeze().cpu().data.numpy()
    print((time.time() - epoch_start_time))
