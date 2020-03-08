# coding: utf8
"""
模型训练和评估代码
"""
from src.fcn.fcns import FCN8
from src.fcn.dataset import VOCDataset
from src.fcn.config import DATASET_ROOT
from src.fcn.evaluator import Evaluator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm

class Trainer:
    def __init__(self, trainSet, network, optimizer, loss, numEpoch, summaryWriter, evaluator, cuda):
        self.trainLoader = DataLoader(trainSet, batch_size=1, shuffle=True)

        self.network = network
        self.optimizer = optimizer
        self.loss = loss

        self.numEpoch = numEpoch
        self.cuda = cuda

        self.summaryWriter = summaryWriter
        self.evaluator = evaluator

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
            pred = F.log_softmax(pred, dim=1)

            loss = self.loss(pred, target)
            loss.backward()
            self.optimizer.step()

            self.summaryWriter.add_scalar("Train/Loss", loss.item(), epoch * len(self.trainLoader) + batchIdx)
            self.summaryWriter.flush()

    def run(self):
        for epoch in range(self.numEpoch):
            self.trainEpoch(epoch)

            # 评估
            meanIou, pixelAcc = self.evaluator.evaluate(self.network)
            self.summaryWriter.add_scalar("Test/PixelAcc", pixelAcc, epoch)
            self.summaryWriter.add_scalar("Test/MeanIOU", meanIou, epoch)
            self.summaryWriter.flush()

if __name__ == '__main__':
    # 数据集
    trainSet = VOCDataset(DATASET_ROOT, True)
    valSet = VOCDataset(DATASET_ROOT, False)

    # 网络和优化器
    network = FCN8(n_class=21)
    network = network.cuda()

    optimizer = torch.optim.SGD(network.parameters(), lr=1e-2, weight_decay=1e-4)
    loss = torch.nn.NLLLoss2d()
    numEpoch = 10

    evaluator = Evaluator(valSet, network, 21, True)

    summaryWriter = SummaryWriter("./tensorboard_log/")
    trainer = Trainer(trainSet, network, optimizer, loss, numEpoch, summaryWriter, evaluator, True)
    trainer.run()







