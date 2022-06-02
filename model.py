import torch.nn as nn
from config import *

def adjust_lr(optimizer, epoch):
    lr_ = configure['lr']
    if epoch > 60:
        lr_ *= 0.5
    elif epoch > 80:
        lr_ *= 0.1
    elif epoch > 100:
        lr_ *= 0.05
    elif epoch > 120:
        lr_ *= 0.01
    elif epoch > 150:
        lr_ *= 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class JorNet(nn.Module):
    def __init__(self):
        super(JorNet, self).__init__() 
        self.layer1 = nn.Sequential(
            BasicBlock(3, 64),
            nn.Dropout(0.5)
            )

        self.layer2 = nn.Sequential(
            BasicBlock(64, 64),
            nn.AvgPool2d(2, 2),
            )

        self.layer3 = nn.Sequential(
            BasicBlock(64, 128),
            nn.Dropout(0.5)
            )

        self.layer4 = nn.Sequential(
            BasicBlock(128, 128),
            nn.AvgPool2d(2, 2),
            )

        self.layer5 = nn.Sequential(
            BasicBlock(128, 256),
            nn.Dropout(0.5)
            )

        self.layer6 = nn.Sequential(
            BasicBlock(256, 256),
            nn.AvgPool2d(8, 8)
            ) 
        
        self.fc1 = nn.Sequential(
            nn.Linear(1 * 1 * 256, 1024),
            nn.Linear(1024, 10)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = x.view(-1, 1 * 1 * 256)  
        x = self.fc1(x)
        return x