'''
Author: MJ.XU
Date: 2021-12-18 23:50:08
LastEditTime: 2022-02-16 16:47:20
LastEditors: MJ.XU
Description: Tech4better
Personal URL: https://www.squirrelled.cn/
'''
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms


class CIFAR10Dataset(object):
    def __init__(self, root_dir, split='train'):
        assert split in ['train', 'test']
        self.root_dir = os.path.join(root_dir, split)

        self.data_list = self.gen_data_list()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.1307] * 3, [0.3081] * 3)
        ])

    def gen_data_list(self, ):
        subfolders = [str(i) for i in range(10)]
        data_list = []
        for sf in subfolders:
            imgs = os.listdir(os.path.join(self.root_dir, sf))
            sub_data_list = [(os.path.join(self.root_dir, sf, img), int(sf))
                             for img in imgs]
            data_list += sub_data_list
        return data_list

    def __getitem__(self, idx):
        path, label = self.data_list[idx]
        im = Image.open(path)

        return {"image": self.transform(im), "label": label}

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    dataset = CIFAR10Dataset(".\\cifar-10-batches-py", 'train')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )
    for d in dataloader:
        print(d)
        break