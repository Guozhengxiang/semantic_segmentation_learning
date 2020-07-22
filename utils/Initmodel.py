import torch
import torch.nn as nn
from utils import config


def init_parameter(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-5
            m.momentum = 0.1
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return model


def load_premodel(model, is_fast=False):
    model = init_parameter(model)
    if is_fast:
        premodel = torch.load(config.fast_model_root)
    else:
        premodel = torch.load(config.ori_model_root)
    local_dict = model.state_dict().copy()
    local_list = list(model.state_dict().keys())
    pre_dict = premodel['model']
    pre_list = list(pre_dict.keys())
    # print(local_list)
    # print(pre_list)
    # parametric switch
    for i in range(len(local_list)):
        if local_list[i].split('.')[0] == 'heads':
            break
        location = pre_list.index(local_list[i])
        # if location != i:
        #     print(local_list[i], pre_list[location])
        local_dict[local_list[i]] = pre_dict[pre_list[location]]
    model.load_state_dict(local_dict)
    return model
