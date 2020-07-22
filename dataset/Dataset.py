import os
import torch
import random
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance

from utils import config


def image2label(im):   # convert RGB images into a label image
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(config.cm2lbl[idx], dtype='int64')


def random_horizontal_flip(img, label, p=0.5):
    if random.random() < p:
        img = F.hflip(img)
        label = F.hflip(label)
    return img, label


def data_crop(data, label, output_size):
    in_w, in_h = data.size
    (out_h, out_w) = output_size
    i = random.randint(0, in_h - out_h)
    j = random.randint(0, in_w - out_w)
    data = F.crop(data, i, j, config.crop_size_h, config.crop_size_w)
    label = F.crop(label, i, j, config.crop_size_h, config.crop_size_w)
    return data, label


def img_enhance(img, label, p=0.5):

    if random.random() < p:  # Color enhancer instance
        img = ImageEnhance.Color(img).enhance(random.uniform(0.5, 1.5))

    if random.random() < p:  # Brightness enhancer instance
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))

    if random.random() < p:  # Contrast enhancer instance
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.5))

    return img, label


def data_enhance(img, label, is_train=True):
    if is_train:
        (img, label) = data_crop(img, label, (config.crop_size_h, config.crop_size_w))
        # (img, label) = random_horizontal_flip(img, label)
        # (img, label) = img_enhance(img, label, p=0.2)
        # label = image2label(label)

    else:
        (img, label) = data_crop(img, label, (config.crop_size_h, config.crop_size_w))
        # label = image2label(label)
    return img, label


class CityDataset(Dataset):
    def __init__(self, split='train', transform=True, is_train=True, ignore_label=255):
        self.root = '/home/froven/桌面/sample'
        self.transform = transform
        self.data_num = []
        self.is_train = is_train
        self.split = split
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

        data_list_file = os.path.join(self.root, '{0}.txt'.format(self.split))
        self.data_num = [id_.strip() for id_ in open(data_list_file)]

    def __len__(self):
        return len(self.data_num)

    def __getitem__(self, i_):
        i_num = self.data_num[i_]
        floder = i_num.split('_')[0]
        img = Image.open(os.path.join(self.root, 'image', self.split, floder, i_num + 'leftImg8bit.png'))
        label = Image.open(os.path.join(self.root, 'label', self.split, floder, i_num + 'gtFine_labelIds.png'))

        if self.transform is True:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if self.is_train:
                (img, label) = data_enhance(img, label, is_train=True)
                img = T.ToTensor()(img)
                img = normalize(img)
                label = np.array(label, dtype=np.int64)
                label = self.transform_label(label)
                label = torch.from_numpy(label)

            else:
                (img, label) = data_enhance(img, label, is_train=False)
                img = T.ToTensor()(img)
                img = normalize(img)
                label = np.array(label, dtype=np.int64)
                label = torch.from_numpy(label)
        return img, label

    def transform_label(self, label):
        label_copy = label.copy()
        ids = np.unique(label_copy)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy


class MyDataset(Dataset):
    def __init__(self, root, split='train', transform=True, is_train=True):
        self.root = root
        self.transform = transform
        self.data_num = []
        self.is_train = is_train
        self.split = split

        data_list_file = os.path.join(self.root, '{0}.txt'.format(self.split))
        self.data_num = [id_.strip() for id_ in open(data_list_file)]

    def __len__(self):
        return len(self.data_num)

    def __getitem__(self, i_):
        i_num = self.data_num[i_]
        img = Image.open(os.path.join(self.root, self.split, 'image', i_num + 'img.png'))
        label = Image.open(os.path.join(self.root, self.split, 'label', i_num + 'label.png')).convert('RGB')

        if self.transform is True:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if self.is_train:
                (img, label) = data_enhance(img, label, is_train=True)
                img = T.ToTensor()(img)
                img = normalize(img)
                # label = image2label(label)
                label = torch.from_numpy(label)

            else:
                # (img, label) = data_enhance(img, label, is_train=False)
                img = T.ToTensor()(img)
                img = normalize(img)
                label = image2label(label)
                label = torch.from_numpy(label)
        return img, label
