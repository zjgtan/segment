# coding: utf8
"""
模型训练和评估代码
"""
from src.fcn.fcns import FCN8
from src.fcn.dataset import VOCDataset
from src.fcn.config import DATASET_ROOT

import torch
from torch.utils.data import DataLoader
import tqdm

class Trainer:
    def __init__(self, trainSet, valSet, network, optimizer, loss, numEpoch, cuda):
        self.trainLoader = DataLoader(trainSet, batch_size=1, shuffle=True)
        self.valLoader = DataLoader(valSet, batch_size=1, shuffle=False)

        self.network = network
        self.optimizer = optimizer
        self.loss = loss

        self.numEpoch = numEpoch
        self.cuda = cuda

    def trainEpoch(self, epoch):
        self.network.train()

        for batchIdx, (data, target) in tqdm.tqdm(enumerate(self.trainLoader),
                                                  total=len(self.trainLoader),
                                                  desc='Train epoch=%d' % epoch, ncols=80, leave=False):
            if self.cuda:
                data = data.cuda()
                target = target.cuda()

            self.optimizer.zero_grad()
            pred = self.network(data)

            loss = self.loss(pred, target)
            loss.backward()
            self.optimizer.step()

    def run(self):
        for epoch in range(self.numEpoch):
            self.trainEpoch(epoch)


if __name__ == '__main__':
    # 数据集
    trainSet = VOCDataset(DATASET_ROOT, True)
    valSet = VOCDataset(DATASET_ROOT, False)

    # 网络和优化器
    network = FCN8(n_class=21)

    network = network.cuda()

    optimizer = torch.optim.SGD(network.parameters(), lr=1e-2, weight_decay=1e-4)
    loss = torch.nn.NLLLoss2d()
    numEpoch = 1

    trainer = Trainer(trainSet, valSet, network, optimizer, loss, numEpoch, True)
    trainer.run()







