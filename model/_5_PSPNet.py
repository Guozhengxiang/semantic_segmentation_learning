import torch
from torch import nn
import torchvision
import torch.nn.functional as F

from model.basemodel.loading_model import resnet_model_load
from model.toolkit import PyramidPoolingModule
from model.toolkit import ConvBnReLU


class PSPNet(nn.Module):

    def __init__(self, num_class):
        super(PSPNet, self).__init__()

        resnet = resnet_model_load(50)
        self.conv1 = resnet.conv1
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        in_channels = 2048
        self.ppm = PyramidPoolingModule(in_channels)

        self.classified_conv = nn.Sequential(
            ConvBnReLU(in_channels * 2, in_channels // 4),
            nn.Conv2d(in_channels//4, num_class, 1)
        )

    def forward(self, x):
        size = x.size()[2:]
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        out = self.classified_conv(self.ppm(x))
        out = F.interpolate(out, size, mode='bilinear', align_corners=True)
        return out


if __name__ == "__main__":
    model = PSPNet(num_class=19)
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

