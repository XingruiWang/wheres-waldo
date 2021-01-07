from model import TemplateMatching
from torch.nn import CrossEntropyLoss, NLLLoss2d
from dataset import Pacman
from torch.utils import data
import torch


def train(net, pacman_loader, criterion):
    net.train()
    for i, data in enumerate(pacman_loader):
        img, template, seudo_label = data
        img, template, seudo_label = img.cuda(), template.cuda(), seudo_label.cuda()
        print(img.shape, template.shape, seudo_label.shape)
        y = net(img, template)
        loss = criterion(y.view(-1), seudo_label.long().view(-1))
        print(loss)



if __name__ == '__main__':  
    net = TemplateMatching(z_dim = 64, output_channel = 512, pretrain = False, num_classes = 1).cuda()
    pacman_set = Pacman(dir='data')
    pacman_loader = data.DataLoader(
        pacman_set,
        batch_size=2,
        num_workers=1,
        pin_memory=True,
        drop_last=True)
    train(net, pacman_loader, criterion = NLLLoss2d())

