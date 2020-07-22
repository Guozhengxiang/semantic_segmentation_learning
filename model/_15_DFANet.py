import torch
import torch.nn as nn
import torch.nn.functional as F

from model.toolkit import ConvBnReLU, SeparableConvBnReLU


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = None

        rep = list()
        rep.append(SeparableConvBnReLU(in_channels, out_channels//4, kernel_size=3, stride=stride, padding=1))
        rep.append(SeparableConvBnReLU(out_channels//4, out_channels//4, kernel_size=3, stride=1, padding=1))
        rep.append(SeparableConvBnReLU(out_channels//4, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu = nn.ReLU(True)
        self.reps = nn.Sequential(*rep)

    def forward(self, x):
        out = self.reps(x)
        if self.skip is not None:
            skip = self.skip(x)
        else:
            skip = x
        out = self.relu(out + skip)
        return out


class DFANetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, stage):
        super(DFANetEncoder, self).__init__()
        rep = list()
        rep.append(Block(in_channels, out_channels, stride=2))
        for i in range(stage - 1):
            rep.append(Block(out_channels, out_channels, stride=1))
        self.reps = nn.Sequential(*rep)

    def forward(self, x):
        return self.reps(x)


class FcAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FcAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1000, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(1000, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, 1000, 1, 1)
        y = self.conv(y)
        return x * y.expand_as(x)


class DFANet(nn.Module):   # Xception
    def __init__(self, num_class):
        super(DFANet, self).__init__()
        self.conv1 = ConvBnReLU(3, 8, kernel_size=3, stride=2, padding=1,
                                has_bn=True, has_relu=True, has_bias=False)

        self.enc2_1 = DFANetEncoder(in_channels=8, out_channels=48, stage=4)
        self.enc3_1 = DFANetEncoder(in_channels=48, out_channels=96, stage=6)
        self.enc4_1 = DFANetEncoder(in_channels=96, out_channels=192, stage=4)
        self.fca_1 = FcAttention(192, 192)

        self.enc2_2 = DFANetEncoder(in_channels=240, out_channels=48, stage=4)
        self.enc3_2 = DFANetEncoder(in_channels=144, out_channels=96, stage=6)
        self.enc4_2 = DFANetEncoder(in_channels=288, out_channels=192, stage=4)
        self.fca_2 = FcAttention(192, 192)

        self.enc2_3 = DFANetEncoder(in_channels=240, out_channels=48, stage=4)
        self.enc3_3 = DFANetEncoder(in_channels=144, out_channels=96, stage=6)
        self.enc4_3 = DFANetEncoder(in_channels=288, out_channels=192, stage=4)
        self.fca_3 = FcAttention(192, 192)

        # fuse to decoder
        self.enc2_1_to_decoder = ConvBnReLU(48, 32, kernel_size=1, stride=1, padding=0,
                                            has_bn=True, has_relu=True, has_bias=False)
        self.enc2_2_to_decoder = ConvBnReLU(48, 32, kernel_size=1, stride=1, padding=0,
                                            has_bn=True, has_relu=True, has_bias=False)
        self.enc2_3_to_decoder = ConvBnReLU(48, 32, kernel_size=1, stride=1, padding=0,
                                            has_bn=True, has_relu=True, has_bias=False)
        self.fca_1_to_decoder = ConvBnReLU(192, 32, kernel_size=1, stride=1, padding=0,
                                           has_bn=True, has_relu=True, has_bias=False)
        self.fca_2_to_decoder = ConvBnReLU(192, 32, kernel_size=1, stride=1, padding=0,
                                           has_bn=True, has_relu=True, has_bias=False)
        self.fca_3_to_decoder = ConvBnReLU(192, 32, kernel_size=1, stride=1, padding=0,
                                           has_bn=True, has_relu=True, has_bias=False)
        self.merge_conv = ConvBnReLU(32, 32, kernel_size=1, stride=1, padding=0,
                                     has_bn=True, has_relu=True, has_bias=False)

        self.classified_conv = nn.Conv2d(32, num_class, 1, 1, bias=False)

    def forward(self, x):
        first_conv = self.conv1(x)
        # first backbone
        enc2_1result = self.enc2_1(first_conv)
        enc3_1result = self.enc3_1(enc2_1result)
        enc4_1result = self.enc4_1(enc3_1result)
        fca_1result = self.fca_1(enc4_1result)
        up_fca_1 = F.interpolate(fca_1result,
                                 enc2_1result.size()[2:],
                                 mode='bilinear',
                                 align_corners=False)
        enc2_2result = self.enc2_2(torch.cat((up_fca_1, enc2_1result), 1))
        enc3_2result = self.enc3_2(torch.cat((enc2_2result, enc3_1result), 1))
        enc4_2result = self.enc4_2(torch.cat((enc3_2result, enc4_1result), 1))
        fca_2result = self.fca_2(enc4_2result)
        up_fca_2 = F.interpolate(fca_2result,
                                 enc2_2result.size()[2:],
                                 mode='bilinear',
                                 align_corners=False)
        enc2_3result = self.enc2_3(torch.cat((up_fca_2, enc2_2result), 1))
        enc3_3result = self.enc3_3(torch.cat((enc2_3result, enc3_2result), 1))
        enc4_3result = self.enc4_3(torch.cat((enc3_3result, enc4_2result), 1))
        fca_3result = self.fca_3(enc4_3result)

        # decoder
        x1 = self.enc2_1_to_decoder(enc2_1result)
        x2 = F.interpolate(self.enc2_2_to_decoder(enc2_2result),
                           x1.size()[2:],
                           mode='bilinear',
                           align_corners=False)
        x3 = F.interpolate(self.enc2_3_to_decoder(enc2_3result),
                           x1.size()[2:],
                           mode='bilinear',
                           align_corners=False)
        x_merge = self.merge_conv(x1 + x2 + x3)
        x_fca1 = F.interpolate(self.fca_1_to_decoder(fca_1result),
                               x_merge.size()[2:],
                               mode='bilinear',
                               align_corners=False)
        x_fca2 = F.interpolate(self.fca_2_to_decoder(fca_2result),
                               x_merge.size()[2:],
                               mode='bilinear',
                               align_corners=False)
        x_fca3 = F.interpolate(self.fca_3_to_decoder(fca_3result),
                               x_merge.size()[2:],
                               mode='bilinear',
                               align_corners=False)
        result = self.classified_conv(x_merge + x_fca1 + x_fca2 + x_fca3)
        result = F.interpolate(result, x.size()[2:], mode='bilinear', align_corners=False)
        return result


if __name__ == "__main__":
    model = DFANet(num_class=19)
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
