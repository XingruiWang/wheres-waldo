import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class HyperNetwork(nn.Module):
    def __init__(self, f_size = 3, z_dim = 64, in_channels = 512, out_channels = 1, feature_size = 16):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.Nz = feature_size * feature_size

        self.linear_in = []
        self.linear_out = nn.Linear(in_features = self.z_dim, out_features = self.out_channels * self.f_size * self.f_size)
        for i in range(self.in_channels):
            self.linear_in.append(nn.Linear(in_features = self.Nz, out_features = self.z_dim))
        self.k = []

    def forward(self, z):
        c, h, w = z.shape # h = w = self.in_size
        print('h = %d'%(int(h)))
        assert(c==self.in_channels)
        assert(h * w == self.Nz)
        for i in range(self.in_channels):
            x = z[:, :, i]
            x = x.flatten()
            a = self.linear_in[i](x)
            k = self.linear_out(a) # (out_channel * f_size * f_size)
            kernel = k.view(self.out_channel,1, self.f_size, self.f_size)
            self.k.append(kernel)
        K = torch.cat(self.k, dim = 1)

        return K # out, in, k, k

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']  
pretrain_url = 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth'


class TemplateMatching(nn.Module):
    def __init__():
        super(TemplateMatching, self).__init__(z_dim = 64, output_channel = 512, pretrain = True)
        self.output_channel = output_channel
        self.z_dim = z_dim
        self.feature_t = make_layers(cfg, batch_norm=True)
        self.feature_x = make_layers(cfg, batch_norm=True)
        self.hyper = HyperNetwork(z_dim = self.z_dim)
        if pretrain:
            self.load_pretrain(pretrain_url)

    def load_pretrain(self, url):
        pretrain_model = torch.load(url)
        pretrain_state_dict = pretrain_model.state_dict() 
        pretrain_keys = pretrain_state_dict.keys()
        state_dict = self.feature_t.state_dict()
        for k in state_dict.keys():
            if k in pretrain_keys:
                state_dict[k] = pretrain_state_dict[k]
                print('load pretrain[%s] ==> model[%s]'%(k, k))
            else:
                print('warning!! fail to load pretrain[%s]'%(k))
        self.feature_t.load_state_dict(state_dict)
        self.feature_x.load_state_dict(state_dict)

    def forward(x, t):
        x = self.feature_x(x)
        t = self.feature_t(t)
        kernel = self.hyper(t)
        x = F.conv2d(x, kernel, padding=1)
        x = nn.BatchNorm2d(self.output_channel)(x)
        x = nn.ReLU(inplace=True)(x)

        return x


if __name__ == '__main__':
    t = torch.randn(32, 3, 512, 512)
    x = torch.randn(32, 3, 512, 512)
    res = TemplateMatching(x, t)
    print(res.shape)

    




