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
import torch
from torch import nn
import torch.nn.functional as F
from model.basemodel.loading_model import vgg_model_load
from model.toolkit import ConvBnReLU


class SegNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(SegNetDecoder, self).__init__()
        layers = list()
        layers.append(ConvBnReLU(in_channels, in_channels // 2))
        for i in range(num_layers - 1):
            layers.append(ConvBnReLU(in_channels//2, in_channels // 2))
        layers.append(ConvBnReLU(in_channels // 2, out_channels))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class SegNet(nn.Module):
    def __init__(self, num_class):
        super(SegNet, self).__init__()
        features = vgg_model_load(16).features
        self.encoder1 = features[:4]
        self.encoder2 = features[5:9]
        self.encoder3 = features[10:16]
        self.encoder4 = features[17:23]
        self.encoder5 = features[24:-1]

        self.decoder5 = SegNetDecoder(512, 512, 1)
        self.decoder4 = SegNetDecoder(512, 256, 1)
        self.decoder3 = SegNetDecoder(256, 128, 1)
        self.decoder2 = SegNetDecoder(128, 64, 0)

        self.classified_conv = nn.Sequential(
            nn.Conv2d(64, num_class, 3, padding=1),
            nn.BatchNorm2d(num_class),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.encoder1(x)
        d1, m1 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)
        x2 = self.encoder2(d1)
        d2, m2 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)
        x3 = self.encoder3(d2)
        d3, m3 = F.max_pool2d(x3, kernel_size=2, stride=2, return_indices=True)
        x4 = self.encoder4(d3)
        d4, m4 = F.max_pool2d(x4, kernel_size=2, stride=2, return_indices=True)
        x5 = self.encoder5(d4)
        d5, m5 = F.max_pool2d(x5, kernel_size=2, stride=2, return_indices=True)

        e5 = self.decoder5(F.max_unpool2d(d5, m5, kernel_size=2, stride=2, output_size=x5.size()))
        e4 = self.decoder4(F.max_unpool2d(e5, m4, kernel_size=2, stride=2, output_size=x4.size()))
        e3 = self.decoder3(F.max_unpool2d(e4, m3, kernel_size=2, stride=2, output_size=x3.size()))
        e2 = self.decoder2(F.max_unpool2d(e3, m2, kernel_size=2, stride=2, output_size=x2.size()))
        e1 = F.max_unpool2d(e2, m1, kernel_size=2, stride=2, output_size=x1.size())

        out = self.classified_conv(e1)
        return out


if __name__ == "__main__":
    model = SegNet(num_class=8)
    # _in = torch.rand((3, 512, 512)).unsqueeze(0).cuda()
    # _out = model(_in)
    # print(_out.size())

    import time
    from torchstat import stat
    # stat(model, (3, 512, 512))
    data = torch.rand((1, 3, 512, 512)).cuda()
    model = model.cuda()
    model.eval()
    epoch_start_time = time.time()
    predict = model(data)
    predict = predict.max(1)[1].squeeze().cpu().data.numpy()
    print((time.time() - epoch_start_time))

