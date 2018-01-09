#############################################
###### This script is made by Doi Kento #####
###### University of Tokyo              #####
#############################################
# add the module path
import sys

# import torch module
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import models, datasets
import segnet, fcn, unet
from torchvision import transforms
from numpy_loader import *
from crossentropy2d import CrossEntropy2d
from focalloss2d import FocalLoss2d

# import python module
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse
from tqdm import tqdm
import datetime

def get_argument():
    # get the argment
    parser = argparse.ArgumentParser(description='Pytorch SegNet')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size for training (default:2)')
    parser.add_argument('--epochs', type=int, default=60, help='number of the epoch to train (default:60)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for training (default:0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default:0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay (default:0.0005)')
    parser.add_argument('--band_num', type=int, default=3, help='number of band (default:3)')
    parser.add_argument('--dropout_ratio', type=float, default=0, help='drop out ratio (default:0)')
    parser.add_argument('model', type=str, help='the net work model')
    parser.add_argument('class_num', type=int, help='number of class')
    parser.add_argument('image_dir_path', type=str, help='the path of image directory (npy)')
    parser.add_argument('GT_dir_path', type=str, help='the path of GT directory (npy)')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='the path of pretrained model')
    parser.add_argument('--weight', type=str, default=None, help='class balancing weight')
    parser.add_argument('gamma', type=float, help='gamma of focal loss (if gamme=0 training is equall to with crossentropy loss')
    parser.add_argument('out_path', type=str, help='output weight path')
    args = parser.parse_args()
    return args


def main(args):
    # Loading the dataset
    trans = transforms.Compose([RandomCrop_Segmentation(640), Flip_Segmentation(), Rotate_Segmentation()])
    val_trans = RandomCrop_Segmentation(640)

    train_dataset = Numpy_SegmentationDataset(os.path.join(args.image_dir_path, 'train'), os.path.join(args.GT_dir_path, 'train'), transform=trans)
    val_dataset = Numpy_SegmentationDataset(os.path.join(args.image_dir_path, 'val'), os.path.join(args.GT_dir_path, 'val'), transform=val_trans)

    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)
    loaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    
    print("Complete the preparing dataset")

    # Loading the class balancing weight
    if args.weight is not None:
        weight = np.load(args.weight)
        weight = torch.from_numpy(weight)
    else:
        weight = None

    # Setting the network
    model = args.model
    if model == 'segnet':
        net = segnet.SegNet(args.band_num , args.class_num, args.dropout_ratio)
    elif model == 'fcn32s':
        net = fcn.FCN32(args.band_num, args.class_num)
    elif model == 'fcn8s':
        net = fcn.FCN8(args.band_num, args.class_num)
    elif model == 'unet':
        net = unet.UNet(args.class_num, args.band_num)
    else:
        print('The model is not added please implement and add')
        sys.exit()
    

    # load the model parameter
    if args.pretrained_model_path:
        print('load the pretraind model.')
        th = torch.load(args.pretrained_model_path)
        net.load_state_dict(th)
    net.cuda()

    # Define a Loss function and optimizer
    criterion = FocalLoss2d(gamma=args.gamma, weight=weight).cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 90)

    # initialize the best accuracy and best model weights
    best_model_wts = net.state_dict()
    best_acc = 0.0

    # initialize the loss and accuracy history
    loss_history = {"train": [], "val":[] }
    acc_history = {"train": [], "val":[] }

    # Train the network
    start_time = time.time()
    for epoch in range(args.epochs):
        scheduler.step()
        print('* ' * 20)
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        print('* ' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            # initialize the runnig loss and corrects
            running_loss = 0.0
            running_corrects = 0

            for i, data in enumerate(loaders[phase]):
                # get the input
                inputs, labels = data
                inputs = inputs.float()
                inputs = inputs / 255.0

                # wrap the in valiables
                if phase == 'train':
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.long().cuda())
                else:
                    inputs = Variable(inputs.cuda(), volatile=True)
                    labels = Variable(labels.long().cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = net(inputs)
                _, preds = torch.max(outputs.data, 1)
                n, c, h, w = labels.size()
                labels = labels.view(n,h,w)
                loss = criterion(outputs, labels)

                # backward + optimize if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statuctics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase] 
            epoch_acc = running_corrects / dataset_sizes[phase] / (640*640)
            loss_history[phase].append(epoch_loss)
            acc_history[phase].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # copy the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = net.state_dict()

    elapsed_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    net.load_state_dict(best_model_wts)
    return net, loss_history, acc_history

if __name__ == '__main__':
    args = get_argument()
    model_weights, loss_history, acc_history = main(args)
    torch.save(model_weights.state_dict(), args.out_path + '/weight.pth')
    training_history = np.zeros((4, args.epochs))
    for i, phase in enumerate(["train", "val"]):
        training_history[i] = loss_history[phase]
        training_history[i+2] = acc_history[phase]
    np.save(args.out_path + '/training_history_{}.npy'.format(datetime.date.today()), training_history)
