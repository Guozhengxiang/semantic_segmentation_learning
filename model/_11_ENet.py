import torch
import torch.nn as nn
import torch.nn.functional as F


class InitialBlock(nn.Module):
    def __init__(self, outchannels):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(3, outchannels - 3, 3, 2, 1, bias=False)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(outchannels)
        self.relu = nn.PReLU()

    def forward(self, x):
        x_conv = self.conv(x)
        x_pool = self.maxpool(x)
        x = torch.cat([x_conv, x_pool], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, dilation=1, asymmetric=False, downsampling=False):
        super(Bottleneck, self).__init__()
        self.downsamping = downsampling
        if downsampling:
            self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)
            self.conv_down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU()
        )

        if downsampling:
            self.conv2 = nn.Sequential(
                nn.Conv2d(inter_channels, inter_channels, 2, stride=2, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.PReLU()
            )
        else:
            if asymmetric:
                self.conv2 = nn.Sequential(
                    nn.Conv2d(inter_channels, inter_channels, (5, 1), padding=(2, 0), bias=False),
                    nn.Conv2d(inter_channels, inter_channels, (1, 5), padding=(0, 2), bias=False),
                    nn.BatchNorm2d(inter_channels),
                    nn.PReLU()
                )
            else:
                self.conv2 = nn.Sequential(
                    nn.Conv2d(inter_channels, inter_channels, 3, dilation=dilation, padding=dilation, bias=False),
                    nn.BatchNorm2d(inter_channels),
                    nn.PReLU()
                )
        self.conv3 = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.1)
        )
        self.act = nn.PReLU()

    def forward(self, x1, x2):
        identity = x2
        out = self.conv1(x1)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.act(out + x2)

        return out


class UpsamplingBottleneck(nn.Module):
    """upsampling Block"""

    def __init__(self, in_channels, inter_channels, out_channels):
        super(UpsamplingBottleneck, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.upsampling = nn.MaxUnpool2d(2)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU(),
            nn.ConvTranspose2d(inter_channels, inter_channels, 2, 2, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU(),
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.1)
        )
        self.act = nn.PReLU()

    def forward(self, x, max_indices):
        out_up = self.conv(x)
        out_up = self.upsampling(out_up, max_indices)

        out_ext = self.block(x)
        out = self.act(out_up + out_ext)
        return out


class ENet(nn.Module):
    """Efficient Neural Network"""

    def __init__(self, num_class):
        super(ENet, self).__init__()
        self.initial = InitialBlock(16)

        self.bottleneck1_0 = Bottleneck(16, 16, 64, downsampling=True)
        self.conv_down1_0 = nn.Sequential(
            nn.Conv2d(16, 64, 1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.bottleneck1_1 = Bottleneck(64, 16, 64)
        self.bottleneck1_2 = Bottleneck(64, 16, 64)
        self.bottleneck1_3 = Bottleneck(64, 16, 64)
        self.bottleneck1_4 = Bottleneck(64, 16, 64)

        self.bottleneck2_0 = Bottleneck(64, 32, 128, downsampling=True)
        self.conv_down2_0 = nn.Sequential(
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.bottleneck2_1 = Bottleneck(128, 32, 128)
        self.bottleneck2_2 = Bottleneck(128, 32, 128, dilation=2)
        self.bottleneck2_3 = Bottleneck(128, 32, 128, asymmetric=True)
        self.bottleneck2_4 = Bottleneck(128, 32, 128, dilation=4)
        self.bottleneck2_5 = Bottleneck(128, 32, 128)
        self.bottleneck2_6 = Bottleneck(128, 32, 128, dilation=8)
        self.bottleneck2_7 = Bottleneck(128, 32, 128, asymmetric=True)
        self.bottleneck2_8 = Bottleneck(128, 32, 128, dilation=16)

        self.bottleneck3_1 = Bottleneck(128, 32, 128)
        self.bottleneck3_2 = Bottleneck(128, 32, 128, dilation=2)
        self.bottleneck3_3 = Bottleneck(128, 32, 128, asymmetric=True)
        self.bottleneck3_4 = Bottleneck(128, 32, 128, dilation=4)
        self.bottleneck3_5 = Bottleneck(128, 32, 128)
        self.bottleneck3_6 = Bottleneck(128, 32, 128, dilation=8)
        self.bottleneck3_7 = Bottleneck(128, 32, 128, asymmetric=True)
        self.bottleneck3_8 = Bottleneck(128, 32, 128, dilation=16)

        self.bottleneck4_0 = UpsamplingBottleneck(128, 16, 64)
        self.bottleneck4_1 = Bottleneck(64, 16, 64)
        self.bottleneck4_2 = Bottleneck(64, 16, 64)

        self.bottleneck5_0 = UpsamplingBottleneck(64, 4, 16)
        self.bottleneck5_1 = Bottleneck(16, 4, 16)

        self.classified_conv = nn.ConvTranspose2d(16, num_class, 2, 2, bias=False)

    def forward(self, x):
        # init
        x = self.initial(x)

        # stage 1
        x_down, max_indices1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        x_down = self.conv_down1_0(x_down)
        x = self.bottleneck1_0(x, x_down)
        x = self.bottleneck1_1(x, x)
        x = self.bottleneck1_2(x, x)
        x = self.bottleneck1_3(x, x)
        x = self.bottleneck1_4(x, x)

        # stage 2
        x_down, max_indices2 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        x_down = self.conv_down2_0(x_down)
        x = self.bottleneck2_0(x, x_down)
        x = self.bottleneck2_1(x, x)
        x = self.bottleneck2_2(x, x)
        x = self.bottleneck2_3(x, x)
        x = self.bottleneck2_4(x, x)
        x = self.bottleneck2_5(x, x)
        x = self.bottleneck2_6(x, x)
        x = self.bottleneck2_7(x, x)
        x = self.bottleneck2_8(x, x)

        # stage 3
        x = self.bottleneck3_1(x, x)
        x = self.bottleneck3_2(x, x)
        x = self.bottleneck3_3(x, x)
        x = self.bottleneck3_4(x, x)
        x = self.bottleneck3_6(x, x)
        x = self.bottleneck3_7(x, x)
        x = self.bottleneck3_8(x, x)

        # stage 4
        x = self.bottleneck4_0(x, max_indices2)
        x = self.bottleneck4_1(x, x)
        x = self.bottleneck4_2(x, x)

        # stage 5
        x = self.bottleneck5_0(x, max_indices1)
        x = self.bottleneck5_1(x, x)

        # out
        x = self.classified_conv(x)
        return x


if __name__ == "__main__":
    model = ENet(num_class=19)
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