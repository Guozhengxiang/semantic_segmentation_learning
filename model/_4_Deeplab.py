import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basemodel.loading_model import resnet_model_load
from model.toolkit import ConvBnReLU


class AsppPlus(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(AsppPlus, self).__init__()
        self.conv0 = ConvBnReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                has_bn=True, has_relu=True, has_bias=False)
        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=1, padding=atrous_rates[0],
                                dilation=atrous_rates[0], has_bn=True, has_relu=True, has_bias=False)
        self.conv2 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=1, padding=atrous_rates[1],
                                dilation=atrous_rates[1], has_bn=True, has_relu=True, has_bias=False)
        self.conv3 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=1, padding=atrous_rates[2],
                                dilation=atrous_rates[2], has_bn=True, has_relu=True, has_bias=False)
        self.conv4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                       has_bn=False, has_relu=True, has_bias=False)
        )

        self.combine_conv = ConvBnReLU(out_channels*5, out_channels, kernel_size=1, stride=1, padding=0,
                                       has_bn=True, has_relu=True, has_bias=False)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = F.interpolate(self.conv4(x), x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x0, x1, x2, x3, x4), dim=1)
        x = self.combine_conv(x)
        return x


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_class):
        super(DeepLabHead, self).__init__()
        self.conv1x1 = ConvBnReLU(in_channels, in_channels, kernel_size=1, stride=1, padding=0,
                                has_bn=True, has_relu=True, has_bias=False)
        self.conv3x3 = ConvBnReLU(in_channels*2, num_class, kernel_size=3, stride=1, padding=1,
                                  has_bn=True, has_relu=True, has_bias=False)

    def forward(self, x1, x2):
        x1 = self.conv1x1(x1)
        x2 = F.interpolate(x2, scale_factor=8, mode='bilinear', align_corners=True)
        out = torch.cat((x1, x2), dim=1)
        out = self.conv3x3(out)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
        return out


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_class):
        super(DeepLabV3Plus, self).__init__()
        self.num_class = num_class
        self.resnet = resnet_model_load(50)
        self.aspp = AsppPlus(2048, 256)
        self.head = DeepLabHead(256, num_class)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        out1 = self.resnet.layer1(x)
        x = self.resnet.layer2(out1)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        out2 = self.aspp(x)
        out = self.head(out1, out2)
        return out


if __name__ == "__main__":
    model = DeepLabV3Plus(num_class=19)
    # _in = torch.rand((2, 3, 512, 512)).cuda()
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
