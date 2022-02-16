'''
Author: MJ.XU
Date: 2021-12-18 22:39:19
LastEditTime: 2022-02-16 16:20:45
LastEditors: MJ.XU
Description: Tech4better
FilePath: \Tutorial-HandWriting-Cls-master\Handwriting_MJXU.py
Personal URL: https://www.squirrelled.cn/
'''
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms


class Handwriting_Dataset_MJXU(object):
    def __init__(self, root_dir, split='train'):
        assert split in ['train', 'test']
        self.root_dir = os.path.join(root_dir, split)

        self.data_list = self.gen_data_list()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.1307] * 3, [0.3081] * 3)
        ])

    def gen_data_list(self, ):
        subfolders = [i for i in os.listdir(self.root_dir)][1:]
        data_list = []
        for sf in subfolders:
            imgs = os.listdir(os.path.join(self.root_dir, sf))[1:]
            sub_data_list = [(os.path.join(self.root_dir, sf, img), sf)
                             for img in imgs]
            data_list += sub_data_list
        # print(data_list)
        return data_list

    def __getitem__(self, idx):
        path, label = self.data_list[idx]
        keys = ["香", "港", "中", "文", "大", "学", "电", "子", "工", "程", "系"]
        values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        dic = dict(zip(keys, values))
        im = Image.open(path).resize((97, 97))

        label = dic[label]
        return self.transform(im), label

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    dataset = Handwriting_Dataset_MJXU("Handwriting_Dataset_MJX", 'train')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )
    for d in dataloader:
        break
    for batch_idx, (data, target) in enumerate(dataloader):
        print(data)
        print(target)
        if batch_idx == 3: break
