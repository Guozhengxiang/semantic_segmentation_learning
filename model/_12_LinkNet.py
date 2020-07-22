import torch
from torch import nn
from model.toolkit import ConvBnReLU, DeconvBnReLU


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                   has_bn=True, has_relu=True, has_bias=False)

        self.conv2 = ConvBnReLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                   has_bn=True, has_relu=False, has_bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)

        return out


def make_encoder(block, in_channels, out_channels):
    layers = []
    for i in range(0, 2):
        if i is 0:
            downsample = ConvBnReLU(in_channels, out_channels, kernel_size=1, stride=2, padding=0,
                                    has_bn=True, has_relu=False, has_bias=False)
            layers.append(block(in_channels, out_channels, 2, downsample))
        else:
            layers.append(block(out_channels, out_channels))

    return nn.Sequential(*layers)


def make_decoder(in_channels, out_channels):
    layers = nn.Sequential(
            ConvBnReLU(in_channels, (in_channels//4), kernel_size=1, stride=1, padding=0,
                       has_bn=True, has_relu=True, has_bias=False),
            DeconvBnReLU((in_channels//4), (in_channels//4), kernel_size=2, stride=2, padding=0,
                         has_bn=True, has_relu=True, inplace=True, has_bias=False),
            ConvBnReLU((in_channels//4), out_channels, kernel_size=1, stride=1, padding=0,
                       has_bn=True, has_relu=True, has_bias=False)
        )
    return layers


class LinkNet(nn.Module):
    def __init__(self, num_class):
        super(LinkNet, self).__init__()
        filters = [64, 128, 256, 512]
        self.feature_scale = 1
        filters = [x // self.feature_scale for x in filters]
        self.inchannels = filters[0]

        # # Encoder
        self.encoder_before = nn.Sequential(
            ConvBnReLU(3, filters[0], kernel_size=7, stride=2, padding=3,
                       has_bn=True, has_relu=True, has_bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        block = ResidualBlock
        self.encoder1 = make_encoder(block=block, in_channels=filters[0], out_channels=filters[0])
        self.encoder2 = make_encoder(block=block, in_channels=filters[0], out_channels=filters[1])
        self.encoder3 = make_encoder(block=block, in_channels=filters[1], out_channels=filters[2])
        self.encoder4 = make_encoder(block=block, in_channels=filters[2], out_channels=filters[3])
        self.avgpool = nn.AvgPool2d(7)

        # # Decoder
        self.decoder4 = make_decoder(in_channels=filters[3], out_channels=filters[2])
        self.decoder3 = make_decoder(in_channels=filters[2], out_channels=filters[1])
        self.decoder2 = make_decoder(in_channels=filters[1], out_channels=filters[0])
        self.decoder1 = make_decoder(in_channels=filters[0], out_channels=filters[0])

        # # Final classify
        self.final_deconv1 = DeconvBnReLU(filters[0], 32//self.feature_scale, kernel_size=2, stride=2, padding=0,
                                          has_bn=True, has_relu=True, inplace=True, has_bias=False)
        self.final_conv2 = ConvBnReLU(32//self.feature_scale, 32//self.feature_scale, kernel_size=3, stride=1, padding=1,
                                      has_bn=True, has_relu=True, has_bias=False)
        self.classified_conv = nn.ConvTranspose2d(32//self.feature_scale, num_class, 2, 2)

    def forward(self, x):
        x = self.encoder_before(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        #
        # # Decoder with Skip Connections
        d4 = self.decoder4(e4)
        d4 = d4 + e3
        d3 = self.decoder3(d4)
        d3 = d3 + e2
        d2 = self.decoder2(d3)
        d2 = d2 + e1
        d1 = self.decoder1(d2)
        #
        # # Final Classification
        x = self.final_deconv1(d1)
        x = self.final_conv2(x)
        x = self.classified_conv(x)
        return x


if __name__ == "__main__":
    model = LinkNet(num_class=19)
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
