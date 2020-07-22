import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.Config import config


class WeightLoss(nn.Module):
    def __init__(self, use_weight=False):
        super(WeightLoss, self).__init__()

        if use_weight:
            '''classname = ['road', 'building', 'fence', 'pole', 'rock', 'tree', 'bush',
                            'terrain', 'sky', 'water', 'person', 'car', 'ship', 'ignore']'''
            weight = torch.FloatTensor([1.0,      # road
                                        1.0,      # building
                                        1.0,      # fence
                                        5.0,      # pole
                                        1.0,      # rock
                                        1.0,      # tree
                                        2.0,      # bush
                                        1.0,      # terrain
                                        1.0,      # sky
                                        1.0,      # water
                                        5.0,      # person
                                        1.0,      # car
                                        5.0])     # ship
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight.cuda(),
                                                       ignore_index=config.ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=config.ignore_label)

    def forward(self, predict, label):
        predict1 = predict.max(1)[1].squeeze().cpu().data.numpy()
        ids = np.unique(predict1)
        return self.criterion(predict, label)


class FocalLoss(nn.Module):
    def __init__(self, num_class, gamma=2, use_weight=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.use_weight = use_weight
        if use_weight:
            weight = torch.FloatTensor(1.0 * np.ones((1, num_class)))
            self.criterion = torch.nn.NLLLoss(weight=weight.cuda(),
                                              ignore_index=config.ignore_label)
        else:
            self.criterion = torch.nn.NLLLoss(ignore_index=config.ignore_label)

    def forward(self, preds, label):
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)
        pred = torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)   # focal_loss

        return self.criterion(pred, label)
