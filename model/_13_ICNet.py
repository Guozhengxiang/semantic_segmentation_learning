import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basemodel.loading_model import resnet_model_load
from model.toolkit import ConvBnReLU, PyramidPoolingModule


class CascadeFeatureFusion(nn.Module):
    def __init__(self, in_channels_low, in_channels_high, out_channels, num_class, is_train=True):
        super(CascadeFeatureFusion, self).__init__()
        self.is_train = is_train
        self.conv_low = ConvBnReLU(in_channels_low, out_channels, kernel_size=3, padding=2, dilation=2,
                                   has_bn=True, has_relu=False, has_bias=False)
        self.conv_high = ConvBnReLU(in_channels_high, out_channels, kernel_size=3, padding=2, dilation=2,
                                    has_bn=True, has_relu=False, has_bias=False)
        self.aux_classify = nn.Conv2d(out_channels, num_class, 1, bias=False)

    def forward(self, x_high, x_low):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)
        if self.is_train is True:
            aux_cls = self.aux_classify(x_low)
            return x, aux_cls
        else:
            return x


class ICNetHead(nn.Module):
    def __init__(self, num_class, is_train=True):
        super(ICNetHead, self).__init__()
        self.is_train = is_train
        self.cff_12 = CascadeFeatureFusion(128, 64, 128, num_class, is_train=is_train)
        self.cff_24 = CascadeFeatureFusion(1024, 512, 128, num_class, is_train=is_train)

        self.classified_conv = nn.Conv2d(128, num_class, 1, bias=False)

    def forward(self, x_1, x_2, x_4):
        aux_classify = []
        if self.is_train is True:
            out_24, aux_cls_24 = self.cff_24(x_2, x_4)  # 1/16
            aux_classify.append(aux_cls_24)
            out_12, aux_cls_12 = self.cff_12(x_1, out_24)
            aux_classify.append(aux_cls_12)
            up_x = F.interpolate(out_12, scale_factor=2, mode='bilinear', align_corners=True)
            up_x = self.classified_conv(up_x)
            aux_classify.append(up_x)
            out = F.interpolate(up_x, scale_factor=4, mode='bilinear', align_corners=True)
            return out, aux_classify  # aux_classify -> 1/16 -> 1/8 -> 1/4
        else:
            out_24 = self.cff_24(x_2, x_4)  # 1/16
            out_12 = self.cff_12(x_1, out_24)
            up_x = F.interpolate(out_12, scale_factor=2, mode='bilinear', align_corners=True)
            up_x = self.classified_conv(up_x)
            out = F.interpolate(up_x, scale_factor=4, mode='bilinear', align_corners=True)
            return out


class Backbone(nn.Module):
    def __init__(self, kinds):
        super(Backbone, self).__init__()
        resnet = resnet_model_load(kinds)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        out = self.layer2(x)
        return out


class ICNet(nn.Module):
    """Image Cascade Network"""

    def __init__(self, num_class, is_train=False):
        super(ICNet, self).__init__()

        self.conv_resolution1 = nn.Sequential(
            ConvBnReLU(3, 32, kernel_size=3, stride=2),
            ConvBnReLU(32, 32, kernel_size=3, stride=2),
            ConvBnReLU(32, 64, kernel_size=3, stride=2)
        )
        self.backbone = Backbone(50)

        self.ppm = PyramidPoolingModule(512)

        self.head = ICNetHead(num_class, is_train=is_train)

        self.__setattr__('exclusive', ['conv_sub1', 'head'])

    def forward(self, x):
        # resolution_1
        x_resolution_1 = self.conv_resolution1(x)

        # resolution_1/2
        x_resolution_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x_resolution_2 = self.backbone(x_resolution_2)

        # resolution_1/4
        x_resolution_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        x_resolution_4 = self.backbone(x_resolution_4)

        # add PyramidPoolingModule
        x_resolution_4 = self.ppm(x_resolution_4)
        outputs = self.head(x_resolution_1, x_resolution_2, x_resolution_4)

        return outputs


if __name__ == "__main__":
    model = ICNet(num_class=19, is_train=False)
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
