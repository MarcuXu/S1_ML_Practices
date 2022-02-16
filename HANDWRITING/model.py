'''
Author: MJ.XU
Date: 2021-11-29 17:16:33
LastEditTime: 2021-12-18 23:28:25
LastEditors: MJ.XU
Description: Tech4better
FilePath: \Tutorial-HandWriting-Cls-master\model.py
Personal URL: https://www.squirrelled.cn/
'''
# pytorch related packages
import torch
import torch.nn as nn
import torch.nn.functional as F


# Model Definition
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, stride=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=40, out_channels=80, kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((3, 3)),
        )
        self.fc1 = nn.Linear(80 * 3 * 3, 500)
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 80 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
