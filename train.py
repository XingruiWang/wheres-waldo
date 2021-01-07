from model import TemplateMatching
from torch.nn import CrossEntropyLoss, NLLLoss2d
from dataset import Pacman
from torch.utils import data
import torch

batch_size = 2

def train(net, pacman_loader, criterion):
    net.train()
    for _ in range(10):
        for i, data in enumerate(pacman_loader):
            img, template, seudo_label = data
            img, template, seudo_label = img.cuda(), template.cuda(), seudo_label.cuda()
            y = net(img, template)
            y_pred = y.permute(0,2,3,1)
            y_pred = y_pred.contiguous().view(-1, 2)
            y_true = seudo_label.long().view(-1)
            loss = criterion(y_pred, y_true)
            loss.backward()
            print(loss.item())

 



if __name__ == '__main__':  
    net = TemplateMatching(z_dim = 64, output_channel = 512, pretrain = False, num_classes = 2).cuda()
    pacman_set = Pacman(dir='data')
    pacman_loader = data.DataLoader(
        pacman_set,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=True)
    train(net, pacman_loader, criterion = CrossEntropyLoss())

