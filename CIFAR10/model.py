'''
Author: MJ.XU
Date: 2021-11-29 17:16:33
LastEditTime: 2021-12-18 22:15:51
LastEditors: MJ.XU
Description: Tech4better
FilePath: \cifar10\model.py
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
        """
		self.conv = 

			2D convolution: input_channel 3, output channel 20, kernel size 3, stride 1
			batch normalization
			ReLU activation
			max pooling: kernel size 3, stride 2

			2D convolution: input_channel 20, output channel 40, kernel size 3, stride 1
			batch normalization
			ReLU activation
			max pooling: kernel size 3, stride 2

			2D convolution: input_channel 40, output channel 80, kernel size 3, stride 1
			batch normalization
			ReLU activation
			max pooling the feature map to: [HxW] = [3x3]
		"""
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
