from model import TemplateMatching
from torch.nn import CrossEntropyLoss, NLLLoss2d
from dataset import Pacman
from torch.utils import data
import torch.optim as optim
import torch
import cv2 as cv
import numpy as np
import os


batch_size = 2
epochs = 100
output = 'output'


def IoU(y, pred):
    pred = np.argmax(pred.permute(0, 2, 3, 1).detach().cpu().numpy(), axis=3)
    y = y.permute(0, 2, 3, 1).detach().cpu().numpy().squeeze(-1)
    if np.sum(y == 1) == 0:
        return max(1 - np.sum(pred == 1) / 144, 1.0)
    intersect = np.sum((y == 1) * (pred == 1))
    union = np.sum((y == 1) + (pred == 1))
    return intersect / union


def train(net, pacman_loader, monster_loader, criterion, optimizer, epochs=50, best_iou = 0):
    net.train()
    running_loss = 0.0
    best_loss = 10
    iou = 0.0
    best_iou = best_iou
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
            iou += IoU(seudo_label, y)

            if i % 5 == 4:
                print('Epoch [%d/%d], iter %d: avg loss = %.3f, avg iou = %.3f' %
                      (epoch + 1, epochs, i + 1, running_loss / 5, iou / 5))
                running_loss = 0.0
                iou = 0.0

            if epoch % 20 == 19:
                vis_img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
                vis_img *= [0.229, 0.224, 0.225]
                vis_img += [0.485, 0.456, 0.406]
                vis_img *= 255
                vis = y.permute(0, 2, 3, 1)[:, :, :, 1].detach().cpu().numpy()

                for b in range(batch_size):
                    vis_mini = vis[b]
                    vis_img_mini = vis_img[b]
                    vis_mini = cv.cvtColor(vis_mini * 255, cv.COLOR_GRAY2BGR)
                    added_image = cv.addWeighted(
                        vis_img_mini[:, :, ::-1], 0.6, vis_mini, 0.4, 0)
                    cv.imwrite('%s/pred/pred-%d.png' %
                               (output, i * batch_size + b), added_image)

        val_loss, val_iou = val(net, monster_loader, criterion, epoch % 20 == 19)

        if val_iou > best_iou:
            checkpoint = {'TemplateMatching': net.state_dict(),
                          'val_loss': val_loss,
                          'best_iou': val_iou
                          }
            torch.save(checkpoint, os.path.join(
                output, "checkpoint", 'model_best.pth'))
            print("IoU improve from %5f to %5f. Save best checkpoint" %
                  (best_iou, val_iou))
            best_iou = val_iou


def val(net, monster_loader, criterion, render = False):
    # net.eval()
    val_loss = 0.0
    iou = 0.0
    for i, data in enumerate(monster_loader):
        img, template, seudo_label = data
        (img, template, seudo_label) = (
            img.cuda(), template.cuda(), seudo_label.cuda())
        y = net(img, template)
        y_pred = y.permute(0, 2, 3, 1)
        y_pred = y_pred.contiguous().view(-1, 2)
        y_true = seudo_label.long().view(-1)
        loss = criterion(y_pred, y_true)
        iou += IoU(seudo_label, y)
        val_loss += loss.item()

        if render:
            vis_img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
            vis_img *= [0.229, 0.224, 0.225]
            vis_img += [0.485, 0.456, 0.406]
            vis_img *= 255
            vis = y.permute(0, 2, 3, 1).detach().cpu().numpy()
            vis = np.argmax(vis, axis = -1).astype(np.uint8)

            for b in range(batch_size):
                vis_mini = vis[b]
                vis_img_mini = vis_img[b]
                vis_mini = cv.cvtColor(vis_mini * 255, cv.COLOR_GRAY2BGR)
                added_image = cv.addWeighted(
                    vis_img_mini[:, :, ::-1].astype(np.uint8), 0.6, vis_mini, 0.4, 0)
                cv.imwrite('%s/pred-monster/pred-%2d.png' %
                           (output, i * batch_size + b), added_image)

    val_loss = val_loss / len(monster_loader)
    iou = iou / len(monster_loader)
    print('Test loss = %.3f, iou = %.3f' % (val_loss, iou))
    return val_loss, iou


if __name__ == '__main__':
    catagory = ['pacman',
                'monster-purple', 'monster-red', 'monster-yellow', 'monster-blue']
    # catagory = ['pacman']
    net = TemplateMatching(z_dim=64,
                           output_channel=512,
                           pretrain='pretrained/vgg11_bn.pth',
                           num_classes=2,
                           freezed_pretrain=True).cuda()
    checkpoint = {}
    checkpoint = torch.load('output/checkpoint/model_best.pth')
    net.load_state_dict(checkpoint['TemplateMatching'])
    best_iou = checkpoint.get('best_iou', 0.0)
    print('best_iou' , best_iou)
    pacman_set = Pacman(dir='data', pad=True,
                        random_template=True, catagory=catagory)
    pacman_loader = data.DataLoader(
        pacman_set,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=True)
    moster_set = Pacman(dir='data', pad=True, catagory=[
                        'monster-blue'], random_template=True)
    monster_loader = data.DataLoader(
        moster_set,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=True)

    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001,
                           betas=(0.9, 0.999), eps=1e-08)
    weights = [1, 30]
    weights = torch.FloatTensor(weights).cuda()
    criterion = CrossEntropyLoss(weight=weights)
    train(net, pacman_loader, monster_loader, criterion=criterion,
          optimizer=optimizer, epochs=epochs, best_iou = best_iou)
    # val(net, monster_loader, criterion)
