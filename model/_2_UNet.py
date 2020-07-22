import torch
import torch.nn as nn
import torch.nn.functional as F
from model.toolkit import ConvBnReLU


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBnReLU(in_channels, out_channels),
            ConvBnReLU(out_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnReLU(in_channels, out_channels),
            ConvBnReLU(out_channels, out_channels),
        )

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, num_class):
        super(UNet, self).__init__()
        self.inc = nn.Sequential(
            ConvBnReLU(3, 64),
            ConvBnReLU(64, 64),
        )
        self.encoder1 = UNetEncoder(64, 128)
        self.encoder2 = UNetEncoder(128, 256)
        self.encoder3 = UNetEncoder(256, 512)
        self.encoder4 = UNetEncoder(512, 1024)
        self.decoder4 = UNetDecoder(1024, 512)
        self.decoder3 = UNetDecoder(512, 256)
        self.decoder2 = UNetDecoder(256, 128)
        self.decoder1 = UNetDecoder(128, 64)
        self.classified_conv = nn.Conv2d(64, num_class, kernel_size=1)

    def forward(self, x):
        x = self.inc(x)
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        out = self.decoder4(x4, x3)
        out = self.decoder3(out, x2)
        out = self.decoder2(out, x1)
        out = self.decoder1(out, x)
        out = self.classified_conv(out)
        return out


if __name__ == "__main__":
    model = UNet(num_class=19)
    # _in = torch.rand((3, 512, 512)).unsqueeze(0).cuda()
    # _out = model(_in)
    # print(_out.size())
    # print(model)

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


