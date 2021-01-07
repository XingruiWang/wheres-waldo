from model import TemplateMatching

import torch


net = TemplateMatching(z_dim = 64, output_channel = 512, pretrain = False).cuda()

def train()
