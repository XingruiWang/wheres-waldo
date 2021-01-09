import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class HyperNetwork(nn.Module):
    def __init__(self,
                 f_size=3,
                 z_dim=64,
                 in_channels=512,
                 out_channels=1,
                 feature_size=16):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.Nz = feature_size * feature_size

        self.linear_in = []
        self.linear_out = nn.Linear(
            in_features=self.z_dim,
            out_features=self.out_channels * self.f_size * self.f_size).cuda()
        for i in range(self.in_channels):
            self.linear_in.append(
                nn.Linear(in_features=self.Nz, out_features=self.z_dim).cuda())
        self.linear_in = nn.ModuleList(self.linear_in)

    def forward(self, z):
        k_list = []
        batch_size, c, h, w = z.shape  # h = w = self.in_size
        assert(c == self.in_channels)
        assert(h * w == self.Nz)
        for i in range(self.in_channels):
            x = z[:, i, :, :]
            x = x.flatten(start_dim=1)
            a = self.linear_in[i](x)
            k = self.linear_out(a)  # (b * out_channel * f_size * f_size)
            kernel = k.view(batch_size, 1, self.out_channels,
                            self.f_size, self.f_size)
            k_list.append(kernel)
        K = torch.cat(k_list, dim=1)

        return K  # b, out, in, k, k


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
    def __init__(self, z_dim=64,
                 output_channel=512,
                 pretrain=None,
                 init_weights=True,
                 num_classes=1,
                 freezed_pretrain=True):
        super(TemplateMatching, self).__init__()
        self.output_channel = output_channel
        self.z_dim = z_dim
        self.feature_t = make_layers(cfg, batch_norm=True)
        self.feature_x = make_layers(cfg, batch_norm=True)
        self.hyper = HyperNetwork(
            z_dim=self.z_dim, out_channels=self.output_channel)
        self.hyper_bn = nn.Sequential(nn.BatchNorm2d(self.output_channel),
                                          nn.ReLU(inplace=True))
        self.final = nn.Conv2d(self.output_channel, num_classes, kernel_size=1)

        if init_weights:
            self._initialize_weights()
        if pretrain:
            self.load_pretrain(pretrain)
        if freezed_pretrain:
            for param in self.feature_t.parameters():
                param.requires_grad = False
            for param in self.feature_x.parameters():
                param.requires_grad = False

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def conv2d(self, x, kernel):
        x = F.conv2d(x, kernel, padding=1)
        x = self.hyper_bn(x)
        return x

    def load_pretrain(self, pretrain):
        pretrain_state_dict = torch.load(pretrain)
        pretrain_keys = pretrain_state_dict.keys()
        state_dict = self.feature_t.state_dict()
        for k in state_dict.keys():
            pretrain_k = 'features.' + k
            if pretrain_k in pretrain_keys:
                state_dict[k] = pretrain_state_dict[pretrain_k]
                print('load pretrain[%s] ==> model[%s]' % (pretrain_k, k))
            else:
                print('fail to load pretrain[%s]' % (k))
        self.feature_t.load_state_dict(state_dict)
        self.feature_x.load_state_dict(state_dict)

    def forward(self, x_img, t):
        batch_size = x_img.shape[0]
        batch_size_t = t.shape[0]
        assert(batch_size == t.shape[0] or t.shape[0] == 1)
        x = self.feature_x(x_img)
        t = self.feature_t(t)
        kernel = self.hyper(t)
        if t.shape[0] == 1:
            x_out = self.conv2d(x, kernel[0])
        else:
            x_out = []
            for b in range(batch_size):
                x_out.append(self.conv2d(x[b].unsqueeze(0), kernel[b]))
            x_out = torch.cat(x_out, dim=0)
        final = self.final(x_out)
        x_out = F.interpolate(final, size=x_img.size()[2:], mode='bicubic')
        x_out = F.softmax(x_out, dim=1)
        return x_out


if __name__ == '__main__':
    t = torch.randn(4, 3, 512, 512).cuda()
    x = torch.randn(4, 3, 512, 512).cuda()
    model = TemplateMatching(
        z_dim=64, output_channel=512, pretrain=False).cuda()
    res = model(x, t)

    print(res.shape)
