from model import TemplateMatching
from torch.nn import CrossEntropyLoss, NLLLoss2d
from dataset import Pacman
from torch.utils import data
import torch.optim as optim
import torch
import cv2 as cv

batch_size = 2
epochs = 100


def train(net, pacman_loader, criterion, optimizer, epochs=50):
    net.train()
    running_loss = 0.0
    for epoch in range(epochs):
        for i, data in enumerate(pacman_loader):
            optimizer.zero_grad()
            img, template, seudo_label = data
            (img, template, seudo_label) = (
                img.cuda(), template.cuda(), seudo_label.cuda())
            y = net(img, template)
            y_pred = y.permute(0, 2, 3, 1)
            y_pred = y_pred.contiguous().view(-1, 2)
            y_true = seudo_label.long().view(-1)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 5 == 4:
                print('Epoch [%d/%d], iter %d: avg loss = %.3f' %
                      (epoch + 1, epochs, i + 1, running_loss / 5))
                running_loss = 0.0
        if epoch % 10 == 0:
            vis_img = img.detach().cpu().permute(0, 2, 3, 1)[0].numpy()
            vis = y.permute(0, 2, 3, 1)[0, :, :, 1].detach().cpu().numpy()
            vis_img *= [0.229, 0.224, 0.225]
            vis_img += [0.485, 0.456, 0.406]
            vis_img *= 255
            vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
            added_image = cv.addWeighted(vis_img[:, :, ::-1], 0.6, vis * 255, 0.4, 0)
            cv.imwrite('output/pred.png', added_image)



if __name__ == '__main__':
    net = TemplateMatching(z_dim=64,
                           output_channel=512,
                           pretrain='pretrained/vgg11_bn.pth',
                           num_classes=2,
                           freezed_pretrain=True).cuda()
    pacman_set = Pacman(dir='data')
    pacman_loader = data.DataLoader(
        pacman_set,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=True)
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    criterion = CrossEntropyLoss()
    train(net, pacman_loader, criterion=criterion, optimizer=optimizer, epochs = epochs)
