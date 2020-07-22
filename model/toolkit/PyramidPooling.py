import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pyramids=[1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()
        self.pyramids = pyramids
        out_channels = in_channels // len(pyramids)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, _input):
        out = _input
        size = _input.size()[2:]
        for bin_size in self.pyramids:
            x = F.adaptive_avg_pool2d(_input, output_size=bin_size)
            x = self.conv(x)
            x = F.interpolate(x, size, mode='bilinear', align_corners=True)
            out = torch.cat([out, x], dim=1)
        return out
