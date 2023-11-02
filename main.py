import os
import argparse
from tqdm.auto import tqdm
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
import torchvision
import warnings
warnings.filterwarnings('ignore')

from quick_cifar import CifarLoader

def construct_rn9():

    class Mul(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight
        def forward(self, x): 
            return x * self.weight

    class Flatten(nn.Module):
        def forward(self, x): 
            return x.view(x.size(0), -1) 

    class Residual(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, x): 
            return x + self.module(x)

    def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, affine=True):
        return nn.Sequential(
                nn.Conv2d(channels_in, channels_out,
                          kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=False),
                nn.BatchNorm2d(channels_out, affine=affine),
                nn.ReLU(inplace=True)
        )   

    NUM_CLASSES = 10
    model = nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(2),
        Residual(nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        nn.Linear(128, NUM_CLASSES, bias=False),
        Mul(0.2),
    )   
    model = model.to(memory_format=torch.channels_last)
    return model

def evaluate(model, loader, tta=False):
    model.eval()
    correct = 0
    with torch.no_grad(), autocast():
        for inputs, labels in loader:
            outs = model(inputs)
            if tta:
                outs += model(inputs.flip(3))
            correct += (outs.argmax(1) == labels).sum().item()
    return correct

def train(train_loader, test_loader, lr=0.5):
    epochs = 64

    model = construct_rn9().cuda()

    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    iters_per_epoch = len(train_loader)
    lr_schedule = np.interp(np.arange(epochs * iters_per_epoch + 1),
                            [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
    scaler = GradScaler()

    losses = []
    for _ in tqdm(range(epochs)):
        for inputs, labels in train_loader:
            with autocast():
                outs = model(inputs)
                loss = F.cross_entropy(outs, labels)
            losses.append(loss.item())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

    print('train correct:', evaluate(model, train_loader))
    print('train loss:', sum(losses[-50:])/50)
    print('test correct:', evaluate(model, test_loader))
    print('test correct w/ tta:', evaluate(model, test_loader, tta=True))
    return model


# this loader just exists to grab its augment method and normalization stats
loader = CifarLoader('/tmp', train=True, batch_size=1, aug=dict(flip=True, translate=2))

steps = 12
r = 2.0 # Ilyas et al. used 2.0
eta = 0.5 # 0.5 seems to always work well
n_noise = 0 # how many extra noised inputs to use.
def pgd_noise(model, inputs, new_labels):

    deltas = torch.zeros_like(inputs, requires_grad=True)
    losses = []
    accs = []
    for _ in range(steps):
        
        inputs_list = [loader.augment(inputs + deltas) for _ in range(n_noise)]
        inputs_list.append(inputs + deltas)
        n_repeat = len(inputs_list)
        inputs_noise = torch.cat(inputs_list)
        
        out = model(inputs_noise)
        loss = F.cross_entropy(out, new_labels.repeat(n_repeat))
        losses.append(loss.item())
        accs.append((out.argmax(1) == new_labels.repeat(n_repeat)).float().mean())
        deltas.grad = None
        loss.backward()

        # a) take gradient step
        gg = deltas.grad
        gg = gg / (gg.reshape(len(gg), -1).norm(dim=1)[:, None, None, None] + 1e-4)
        deltas.data -= eta * r * gg
        
        # b) project back to L2-ball of radius r
        rr = deltas.data.reshape(len(deltas), -1).norm(dim=1)
        mask = (rr > r)
        deltas.data[mask] *= r / rr[mask][:, None, None, None]

        # c) move back to valid pixel space [0, 1]
        new_inputs = (inputs + deltas.data)
        mean, std = loader.mean, loader.std
        new_inputs = ((new_inputs * std + mean).clip(0, 1) - mean) / std

    return new_inputs


def main():
    # 1. Train a normal model.
    train_loader = CifarLoader('/tmp', train=True, batch_size=500, aug=dict(flip=True, translate=2))
    test_loader = CifarLoader('/tmp', train=False, batch_size=1000)
    model = train(train_loader, test_loader, lr=0.5)

    # 2. Generate new dataset of adversarial examples.
    for p in model.parameters():
        p.require_grad = False

    loader = CifarLoader('/tmp', train=True, batch_size=500, shuffle=False)
    new_xx = []
    new_yy = []
    for inputs, labels in tqdm(loader):
        new_labels = torch.randint(10, size=labels.shape).cuda()
        new_inputs = pgd_noise(model, inputs, new_labels)
        new_xx.append(new_inputs)
        new_yy.append(new_labels)
    new_images = torch.cat(new_xx)
    new_targets = torch.cat(new_yy)

    max_r = (new_images - train_loader.images).reshape(len(new_images), -1).norm(dim=1).max()
    print('max r:', max_r.cpu()) # should be at most the desired radius
    train_loader.images = new_images
    train_loader.targets = new_targets

    # measure the attack success rate with and without applying the data augmentation
    aug = train_loader.aug
    train_loader.aug = {}
    print('attack success rate:', evaluate(model, train_loader) / len(new_images))
    train_loader.aug = aug
    print('attack success rate (with augmentation):', evaluate(model, train_loader) / len(new_images))

    # 3. Train another model using the new dataset.
    train(train_loader, test_loader, lr=0.1)

if __name__ == '__main__':
    main()

