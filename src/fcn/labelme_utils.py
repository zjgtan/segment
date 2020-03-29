# coding: utf8
"""
labelme工具类
"""
import numpy as np
from PIL import Image

class LabelmeUtils:
    def __init__(self):
        self.classes = ['background',
                        'product']
        self.classLabelIndex = dict(zip(self.classes,
                                         [idx for idx in range(len(self.classes))]))

        self.colors = [[0,0,0],
                       [128,0,0]]

        # 通过class label查颜色索引
        self.classColorIndex = [self.getColorIndex(color) for color in self.colors]
        # 通过颜色索引查具体的颜色值
        self.colorIndex = dict(zip(self.classColorIndex, self.colors))
        self.colorIndex2Class = dict(zip(self.classColorIndex, [idx for idx in range(len(self.classColorIndex))]))

        self.cm2lbl = self.getColorMap2LabelIndex()

    def getColorIndex(self, color):
        return (color[0] * 256 + color[1]) * 256 + color[2]

    def getColorMap2LabelIndex(self):
        cm2lbl = np.zeros(256 ** 3)  # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
        for i, cm in enumerate(self.colors):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i  # 建立索引

        return cm2lbl

    def label2Image(self, label):
        image = np.zeros((label.shape[0], label.shape[1], 3),
                         dtype="uint8")

        for x in range(label.shape[0]):
            for y in range(label.shape[1]):
                image[x, y] = self.colorIndex[self.classColorIndex[label[x, y]]]

        #image = Image.fromarray(image.astype('uint8')).convert('RGB')
        return image

    def image2Label(self, image):
        data = np.asanyarray(image, dtype="int64")
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        label = np.array(self.cm2lbl[idx], dtype='int64')

        return label
