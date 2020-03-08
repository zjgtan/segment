# coding: utf8
"""
测试评估器
"""
import unittest
from src.fcn.config import DATASET_ROOT
from src.fcn.dataset import VOCDataset
from torch.utils.data.dataloader import DataLoader
from src.fcn.fcns import FCN8
from src.fcn.evaluator import Evaluator

class TestEvaluator(unittest.TestCase):
    def testEvaluate(self):
        dataset = VOCDataset(DATASET_ROOT, True)

        # 网络和优化器
        network = FCN8(n_class=21)
        network = network.cuda()

        evalator = Evaluator(dataset, network, 21, True)
        ious, pixelAcc = evalator.evaluate()
        print(ious)
        print(pixelAcc)

if __name__ == '__main__':
    unittest.main()

