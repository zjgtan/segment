# coding: utf8
import os
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import torch
import cv2

classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']

colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]

class VOCDataset(Dataset):
    def __init__(self, rootDir, isTrain):
        self.imageFileList, self.labelFileList = self.load(rootDir, isTrain)
        self.cm2lbl = self.getColorMap2LabelIndex()

        self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

        self.imageTransFormer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def getColorMap2LabelIndex(self):
        cm2lbl = np.zeros(256 ** 3)  # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i  # 建立索引

        return cm2lbl

    def load(self, rootDir, isTrain):
        fileName = os.path.join(rootDir, 'ImageSets', 'Segmentation', ('train.txt' if isTrain else 'val.txt'))
        with open(fileName) as file:
            fileNames = [line.rstrip() for line in file]
            imageFileList = [os.path.join(rootDir, 'JPEGImages', line.rstrip() + '.jpg') for line in fileNames]
            labelFileList = [os.path.join(rootDir, 'SegmentationClass', line.rstrip() + '.png') for line in fileNames]

        print("加载%d张图片" % (len(imageFileList)))

        return imageFileList, labelFileList

    def __getitem__(self, idx):
        image = cv2.imread(self.imageFileList[idx])
        label = cv2.imread(self.labelFileList[idx])

        image = self.imageTransform(image)
        label = self.labelTransform(label)

        return image, label

    def __len__(self):
        return len(self.imageFileList)

    def imageTransform(self, img):
        # 使得数据集能被32整除
        img = self.imageTransFormer(img)
        return img

    def labelTransform(self, label):
        data = np.array(label, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        label = np.array(self.cm2lbl[idx], dtype='int64')  # 根据索引得到 label 矩阵
        return torch.from_numpy(label)




