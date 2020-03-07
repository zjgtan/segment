# coding: utf8
"""
数据集测试
"""
import unittest
from src.fcn.dataset import VOCDataset
from src.fcn.config import DATASET_ROOT

class TestDataset(unittest.TestCase):
    def test(self):
        dataset = VOCDataset(DATASET_ROOT, True)
        image, label = dataset[1]



if __name__ == '__main__':
    unittest.main()
