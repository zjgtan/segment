# coding: utf8

import unittest
from torch.utils.data import DataLoader
from src.fcn.fcns import FCN8
from src.fcn.dataset import VOCDataset
from src.fcn.config import DATASET_ROOT

class TestFCNs(unittest.TestCase):
    def testForward(self):
        network = FCN8(n_class=21)

        dataset = VOCDataset(DATASET_ROOT, True)
        self.trainLoader = DataLoader(dataset, batch_size=1, shuffle=True)
        for data, label in self.trainLoader:
            preds = network(data)
            break

if __name__ == '__main__':
    unittest.main()