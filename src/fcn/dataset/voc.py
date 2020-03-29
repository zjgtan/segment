# coding: utf8
import os
import numpy as np
import logging
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import torch

class VOC(Dataset):
    """
    VOC数据集
    """
    def __init__(self, data_dir, is_train):
        self.image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.n_classes = 21

        self.image_file_list, self.label_file_list = self.load_image_list(data_dir, is_train)

    def get_pascal_labels(self):
        return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )

    def load_image_list(self, data_dir, is_train):
        file_image_list = os.path.join(data_dir,
                                       'ImageSets',
                                       'Segmentation',
                                       ('train.txt' if is_train else 'val.txt'))

        with open(file_image_list) as file:
            fileNames = [line.rstrip() for line in file]
            imageFileList = [os.path.join(data_dir, 'JPEGImages', line.rstrip() + '.jpg')
                             for line in fileNames]
            labelFileList = [os.path.join(data_dir, 'SegmentationClass', line.rstrip() + '.png')
                             for line in fileNames]

        return imageFileList, labelFileList

    def __getitem__(self, idx):
        image = Image.open(self.image_file_list[idx]).convert("RGB")
        label = Image.open(self.label_file_list[idx]).convert("RGB")

        image = self.image_transform(image)

        label = self.colormap_to_label(label)
        label = torch.LongTensor(label)

        return image, label

    def get_ori_image(self, idx):
        return np.array(Image.open(self.image_file_list[idx]).convert("RGB")).astype("uint8")

    def image_transform(self, img):
        return self.image_transformer(img)

    def colormap_to_label(self, mask):
        mask = np.array(mask).astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def label_to_colormap(self, label_mask):
        label_mask = np.squeeze(label_mask)
        label_colours = self.get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b

        return rgb

    def __len__(self):
        return len(self.image_file_list)
