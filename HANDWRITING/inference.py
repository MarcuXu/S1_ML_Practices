'''
Author: MJ.XU
Date: 2021-12-18 23:58:36
LastEditTime: 2022-02-16 16:20:15
LastEditors: MJ.XU
Description: Tech4better
Personal URL: https://www.squirrelled.cn/
'''
# basic packages
import os
import os.path as osp
# third-party packages
import pyprind
import glog as log
# pytorch related packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as datasets
import argparse
# model definition
from model import Net
# PIL image
from PIL import Image

# Training Part
# Please fill the training part based on the given model/dataloader/optimizer/criterion


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--resume',
                        type=str,
                        default='char_cnn.pt',
                        help="Model Path.")
    args = parser.parse_args()

    # loading trained models
    print("Initializing model: {}".format(args.resume))
    trained_model = torch.load(args.resume)
    classes = trained_model["classes"]
    model = Net(num_classes=len(classes))
    model.load_state_dict(trained_model["state_dict"])

    image_path = input("Image Path (q to exit): ")
    transforms = T.Compose([
        T.Resize([96, 96]),
        T.ToTensor(),
        T.Normalize(mean=[
            0.5,
            0.5,
            0.5,
        ], std=[1., 1., 1.]),
    ])
    while image_path:
        if image_path == "q" or not image_path:
            return 0
        if not os.path.exists(image_path):
            print("File not Found: {}".format(image_path))
            image_path = input("Image Path (q to exit): ")
            continue
        img = Image.open(image_path)
        img = transforms(img).unsqueeze(0)
        output = model(img)
        pred = output.max(1, keepdim=False)[1][0].item(
        )  # get the index of the max log-probability
        print("{}: {}".format(image_path, classes[pred]))
        image_path = input("Image Path (q to exit): ")


if __name__ == '__main__':
    main()
