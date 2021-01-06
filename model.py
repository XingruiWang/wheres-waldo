import numpy as np
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
        self.linear_out = nn.Linear(in_features = self.z_dim, out_features = self.out_channels * self.f_size * self.f_size).cuda()
        for i in range(self.in_channels):
            self.linear_in.append(nn.Linear(in_features = self.Nz, out_features = self.z_dim).cuda())
        self.k = []

    def forward(self, z):
        batch_size, c, h, w = z.shape # h = w = self.in_size
        assert(c==self.in_channels)
        assert(h * w == self.Nz)
        for i in range(self.in_channels):
            x = z[:, i, :, :]
            x = x.flatten(start_dim=1)
            a = self.linear_in[i](x)
            k = self.linear_out(a) # (b * out_channel * f_size * f_size)
            kernel = k.view(batch_size, self.out_channels,1, self.f_size, self.f_size)
            self.k.append(kernel)
        K = torch.cat(self.k, dim = 2)

        return K # b, out, in, k, k

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
    def __init__(self, z_dim = 64, output_channel = 512, pretrain = False):
        super(TemplateMatching, self).__init__()
        self.output_channel = output_channel
        self.z_dim = z_dim
        self.feature_t = make_layers(cfg, batch_norm=True)
        self.feature_x = make_layers(cfg, batch_norm=True)
        self.hyper = HyperNetwork(z_dim = self.z_dim, out_channels = self.output_channel)
        if pretrain:
            self.load_pretrain(pretrain_url)

    def conv2d(self, x, kernel):
        x = F.conv2d(x, kernel, padding=1)
        x = nn.BatchNorm2d(self.output_channel).cuda()(x)
        x = nn.ReLU(inplace=True).cuda()(x)  
        return x      

    def load_pretrain(self, url):
        pretrain_model = torch.load_state_dict_from_url(url)
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

    def forward(self, x, t):
        batch_size = x.shape[0]
        x = self.feature_x(x)
        t = self.feature_t(t)
        kernel = self.hyper(t)
        x_out = []
        for b in range(batch_size):
            x_out.append(self.conv2d(x[b].unsqueeze(0), kernel[b]))
        x_out = torch.cat(x_out, dim = 0)
        return x_out


if __name__ == '__main__':
    t = torch.randn(8, 3, 512, 512).cuda()
    x = torch.randn(8, 3, 512, 512).cuda()
    model = TemplateMatching(z_dim = 64, output_channel = 32, pretrain = False).cuda()
    res = model(x, t)

    print(res.shape)

    




