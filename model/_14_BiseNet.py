import torch
import torch.nn as nn
import torch.nn.functional as F

from model.toolkit import ConvBnReLU
from model.basemodel.loading_model import resnet_model_load


class AttentionRefinement(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionRefinement, self).__init__()
        self.conv_3x3 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                   has_bn=True, has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnReLU(out_channels, out_channels, kernel_size=1, stride=1, padding=0,
                       has_bn=False, has_relu=False, has_bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        fm = self.conv_3x3(x)
        fm_se = self.channel_attention(fm)
        fm = fm * fm_se
        return fm


class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1):
        super(FeatureFusion, self).__init__()
        self.conv_1x1 = ConvBnReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                   has_bn=True, has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnReLU(out_channels, out_channels // reduction, kernel_size=1, stride=1, padding=0,
                       has_bn=False, has_relu=True, has_bias=False),
            ConvBnReLU(out_channels // reduction, out_channels, kernel_size=1, stride=1, padding=0,
                       has_bn=False, has_relu=False, has_bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.channel_attention(fm)
        output = fm + fm * fm_se
        return output


class BiSeNetHead(nn.Module):
    def __init__(self, in_channels, num_class, scale, is_aux=False):
        super(BiSeNetHead, self).__init__()
        if is_aux:
            self.conv_3x3 = ConvBnReLU(in_channels, 256, kernel_size=3, stride=1, padding=1,
                                       has_bn=True, has_relu=True, has_bias=False)
        else:
            self.conv_3x3 = ConvBnReLU(in_channels, 64, kernel_size=3, stride=1, padding=1,
                                       has_bn=True, has_relu=True, has_bias=False)
        if is_aux:
            self.classified_conv = nn.Conv2d(256, num_class, kernel_size=1, stride=1, padding=0)
        else:
            self.classified_conv = nn.Conv2d(64, num_class, kernel_size=1, stride=1, padding=0)
        self.scale = scale

    def forward(self, x):
        fm = self.conv_3x3(x)
        output = self.classified_conv(fm)
        if self.scale > 1:
            output = F.interpolate(output, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return output


class SpatialPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialPath, self).__init__()
        mid_channel = 64
        self.conv_7x7 = ConvBnReLU(in_channels, mid_channel, kernel_size=7, stride=2, padding=3,
                                   has_bn=True, has_relu=True, has_bias=False)
        self.conv_3x3_1 = ConvBnReLU(mid_channel, mid_channel, kernel_size=3, stride=2, padding=1,
                                     has_bn=True, has_relu=True, has_bias=False)
        self.conv_3x3_2 = ConvBnReLU(mid_channel, mid_channel, kernel_size=3, stride=2, padding=1,
                                     has_bn=True, has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnReLU(mid_channel, out_channels, kernel_size=1, stride=1, padding=0,
                                   has_bn=True, has_relu=True, has_bias=False)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        output = self.conv_1x1(x)
        return output


class ContextPath(nn.Module):      # resnet18
    def __init__(self):
        super(ContextPath, self).__init__()
        resnet = resnet_model_load(18)

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

        blocks = []
        x = self.layer1(x)
        blocks.append(x)
        x = self.layer2(x)
        blocks.append(x)
        x = self.layer3(x)
        blocks.append(x)
        x = self.layer4(x)
        blocks.append(x)
        return blocks


class BiSeNet(nn.Module):
    def __init__(self, num_class, is_train=False):
        super(BiSeNet, self).__init__()
        # self.first_pooling = nn.MaxPool2d(2, 2)
        self.context_path = ContextPath()
        self.business_layer = []
        self.is_train = is_train
        self.spatial_path = SpatialPath(3, 128)
        conv_channel = 128
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnReLU(512, conv_channel, kernel_size=1, stride=1, padding=0,
                       has_bn=False, has_relu=True, has_bias=False)
        )

        # stage = [512, 256, 128, 64]
        self.arms = nn.Sequential(
            AttentionRefinement(512, conv_channel),
            AttentionRefinement(256, conv_channel)
        )
        self.refines = nn.Sequential(
            ConvBnReLU(conv_channel, conv_channel, kernel_size=3, stride=1, padding=1,
                       has_bn=True, has_relu=True, has_bias=False),
            ConvBnReLU(conv_channel, conv_channel, kernel_size=3, stride=1, padding=1,
                       has_bn=True, has_relu=True, has_bias=False)
        )
        self.heads = nn.Sequential(
            BiSeNetHead(conv_channel, num_class, 16, True),
            BiSeNetHead(conv_channel, num_class, 8, True),
            BiSeNetHead(conv_channel * 2, num_class, 8, False)
        )
        self.ffm = FeatureFusion(conv_channel * 2, conv_channel * 2, 1)

    def forward(self, data, label=None):
        # data = self.first_pooling(data)
        spatial_out = self.spatial_path(data)

        context_blocks = self.context_path(data)
        context_blocks.reverse()

        global_context = self.global_context(context_blocks[0])
        global_context = F.interpolate(global_context,
                                       size=context_blocks[0].size()[2:],
                                       mode='bilinear', align_corners=True)

        last_fm = global_context
        pred_out = []

        for i, (fm, arm, refine) in enumerate(zip(context_blocks[:2], self.arms,
                                                  self.refines)):
            fm = arm(fm)
            fm += last_fm
            last_fm = F.interpolate(fm, size=(context_blocks[i + 1].size()[2:]),
                                    mode='bilinear', align_corners=True)
            last_fm = refine(last_fm)
            pred_out.append(last_fm)
        context_out = last_fm

        concate_fm = self.ffm(spatial_out, context_out)
        # concate_fm = self.heads[-1](concate_fm)
        pred_out.append(concate_fm)

        if self.is_train:      # aux1, aux2 , main
            return self.heads[0](pred_out[0]), self.heads[1](pred_out[1]), self.heads[-1](pred_out[2])

        return F.log_softmax(self.heads[-1](pred_out[-1]), dim=1)


if __name__ == "__main__":
    model = BiSeNet(num_class=19, is_train=False)
    # _in = torch.rand((2, 3, 512, 512)).cuda()
    # _out = model(_in)
    # print(_out.size())

    import time
    from torchstat import stat
    stat(model, (3, 512, 512))
    # data = torch.rand((1, 3, 512, 512)).cuda()
    # model = model.cuda()
    # model.eval()
    # epoch_start_time = time.time()
    # predict = model(data)
    # predict = predict.max(1)[1].squeeze().cpu().data.numpy()
    # print((time.time() - epoch_start_time))
