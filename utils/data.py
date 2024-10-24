import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

import os
from PIL import Image, ImageOps, ImageFilter
import os.path as osp
import sys
import random
import shutil


class IRSTD_Dataset(Data.Dataset):
    def __init__(self, args, mode,name):
        
        dataset_dir = args.dataset_dir

        # NUDT or NUAA数据集
        if mode == 'train':
             txtfile = 'train_'+name+'.txt'
        elif mode == 'val':
             txtfile = 'test_'+name+'.txt'

        self.list_dir = osp.join(dataset_dir, txtfile)
        self.imgs_dir = osp.join(dataset_dir, 'image')
        self.label_dir = osp.join(dataset_dir, 'mask')

        # 对文本文件中的每一行进行迭代，去除行末的换行符，并将其添加到 self.names 列表中
        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]


        self.mode = mode
        self.crop_size = args.crop_size
        self.base_size = args.base_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __getitem__(self, i):

        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')

        label_path = osp.join(self.label_dir, name+'.png')

        #label_path = osp.join(self.label_dir, name + '_pixels0.png')

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        # 训练和验证数据增强操作
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        
        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        # 随机水平翻转：以50%的概率对图像和掩码进行水平翻转，增加数据的多样性。
        # 随机缩放：随机选择一个长边的长度，在保持长宽比的情况下缩放图像和掩码。
        # 填充裁剪：如果缩放后的图像较小，对其进行填充，使其大小达到指定的裁剪大小。
        # 随机裁剪：从填充后的图像中随机裁剪出指定大小的区域。
        # 高斯模糊：以50%的概率对图像进行高斯模糊处理，用于模拟图像的自然模糊。
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img, mask


    def _testval_sync_transform(self, img, mask):
        # 这段代码是一个图像和掩码的变换函数，它将输入的图像和掩码都调整到相同的大小，大小由 self.base_size 指定。
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask
