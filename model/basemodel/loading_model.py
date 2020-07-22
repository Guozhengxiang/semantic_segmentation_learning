import torch.nn as nn
import torchvision
from torchvision.models import VGG


def vgg_model_load(kind):
    if kind == 11:
        vgg = torchvision.models.vgg11(pretrained=False)
    if kind == 16:
        vgg = torchvision.models.vgg16(pretrained=False)
    if kind == 19:
        vgg = torchvision.models.vgg19(pretrained=False)

    # for param in vgg.parameters():
    #     param.requires_grad = False

    return vgg


def resnet_model_load(kind, downsampling_ratio=1):

    if kind == 18:
        resnet = torchvision.models.resnet18(pretrained=False)
    if kind == 34:
        resnet = torchvision.models.resnet34(pretrained=False)
    if kind == 50:
        resnet = torchvision.models.resnet50(pretrained=False)
    if kind == 101:
        resnet = torchvision.models.resnet101(pretrained=False)
    if kind == 152:
        resnet = torchvision.models.resnet152(pretrained=True)

    # for param in resnet.parameters():
    #     param.requires_grad = False

    return resnet


if __name__ == "__main__":
    model = resnet_model_load(50)
    print(model)
