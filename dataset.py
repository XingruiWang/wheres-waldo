import os

import cv2
import numpy as np
import random
from ccoeff import template_matching

import torch
from torch.nn import functional as F
from torch.utils import data


class Pacman(data.Dataset):
    def __init__(self,
                 dir='data',
                 catagory=['pacman'],
                 crop=True,
                 base_size=512,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        self.base_size = base_size
        self.crop = crop
        self.catagory = catagory

        self.mean = mean
        self.std = std
        self.dir = dir

        self.source_files = [os.path.join(self.dir, "source", f) for f in os.listdir(
            os.path.join(self.dir, "source"))]
        self.templates_dir = dict(
            zip(catagory, [os.path.join(self.dir, 'templates', c) for c in catagory]))
        self.template_file = dict(zip(catagory, [os.path.join(
            self.dir, 'templates', c, c + '1.png') for c in catagory]))

    def __len__(self):
        return len(self.source_files)

    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def pad_to_img(self, img, template):
        h1, w1, c = img.shape
        h2, w2, c = template.shape
        top = int((h1 - h2) / 2)
        left = int((w1 - w2) / 2)
        bottom = h1 - top
        right = w1 - left
        pad_template = cv2.copyMakeBorder(
            template, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT)
        return pad_template

    def __getitem__(self, i):
        img = cv2.imread(self.source_files[i])
        if self.crop:
            img = img[:180, :, :]
        seudo_label = template_matching(img, self.templates_dir[self.catagory[0]],
                                        vis=False, return_ori=False)
        template = cv2.imread(self.template_file[self.catagory[0]])
        template = self.pad_to_img(img, template)

        img = self.input_transform(img)
        template = self.input_transform(template)

        img = cv2.resize(img, (self.base_size, self.base_size),
                         interpolation=cv2.INTER_NEAREST)
        img = img.transpose((2,0,1))
        template = cv2.resize(
            template, (self.base_size, self.base_size), interpolation=cv2.INTER_NEAREST).transpose((2,0,1))
        seudo_label = cv2.resize(seudo_label, (self.base_size, self.base_size),
                         interpolation=cv2.INTER_NEAREST)[:, :, np.newaxis]
        seudo_label = seudo_label.transpose((2,0,1)).astype(np.float32)
        return img, template, seudo_label


if __name__ == '__main__':
    pacman = Pacman()
    dataset_iter = iter(pacman)
    img, template, label = next(dataset_iter)
    print(img.shape, template.shape, label.shape)