# coding: utf8
"""
模型训练和评估代码
"""
from src.fcn.fcns import FCN8s
from src.fcn.dataset.voc import VOC
from src.fcn.evaluator import Evaluator

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import tqdm
from PIL import Image
import random

class Trainer:
    def __init__(self, train_set, val_set, batch_size, network, optimizer, loss, num_epoch, summary_writer, use_cuda):
        self.train_set = train_set
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)

        self.val_set = val_set
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False)

        self.network = network
        self.optimizer = optimizer
        self.loss = loss

        self.num_epoch = num_epoch
        self.use_cuda = use_cuda

        self.summary_writer = summary_writer

    def train_epoch(self, epoch):
        self.network.train()

        for batchIdx, (data, target) in tqdm.tqdm(enumerate(self.train_loader),
                                                  total=len(self.train_loader),
                                                  desc='Train epoch=%d' % epoch, ncols=80, leave=False):
            if self.use_cuda:
                data = data.cuda()
                target = target.cuda()

            self.optimizer.zero_grad()
            pred = self.network(data)
            pred = F.log_softmax(pred, dim=1)

            loss = self.loss(pred, target)
            loss.backward()
            self.optimizer.step()

            self.summary_writer.add_scalar("Train/Loss", loss.item(), epoch * len(self.train_loader) + batchIdx)
            self.summary_writer.flush()

    def predict(self, image, network):
        preds = network(torch.unsqueeze(image, 0)).squeeze()
        preds = preds.cpu().detach().numpy().transpose(1, 2, 0)
        preds = np.argmax(preds, 2)


    def run(self):
        for epoch in range(self.num_epoch):
            self.train_epoch(epoch)

            '''
            # 评估
            meanIou, pixelAcc = self.evaluator.evaluate(self.network)
            self.summaryWriter.add_scalar("Test/PixelAcc", pixelAcc, epoch)
            self.summaryWriter.add_scalar("Test/MeanIOU", meanIou, epoch)
            self.summaryWriter.flush()
            '''

            '''
            # 预测结果
            idx = random.randint(0, len(self.trainSet) - 1)
            transImage, label = self.trainSet[idx]
            transImage = transImage.cuda()
            predImage = self.predict(transImage, self.network)

            image = np.asarray(Image.open(self.trainSet.imageFileList[idx]).convert("RGB"), dtype="uint8")
            labelImage = np.asarray(Image.open(self.trainSet.labelFileList[idx]).convert("RGB"), dtype="uint8")

            self.summaryWriter.add_image("input", image, epoch, dataformats="HWC")
            self.summaryWriter.add_image("pred", predImage, epoch, dataformats="HWC")
            self.summaryWriter.add_image("label", labelImage, epoch, dataformats="HWC")
            self.summaryWriter.flush()
            '''


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--d_train_dir', default=None)
    arg_parser.add_argument('--d_val_dir', default=None)
    arg_parser.add_argument('--d_common_data_dir', default=None)
    arg_parser.add_argument('--d_pre_model_dir', default=None)
    arg_parser.add_argument('--d_model_dir', default=None)
    arg_parser.add_argument('--d_summary_dir', default="./run")
    arg_parser.add_argument('--d_result_dir', default=None)

    arg_parser.add_argument("--data_dir", default="E:\\dataset\\VOCdevkit\\VOC2012")
    arg_parser.add_argument("--use_cuda", type=bool, default=True)
    arg_parser.add_argument("--num_epoch", type=int, default=1)
    arg_parser.add_argument("--batch_size", type=int, default=1)
    arg_parser.add_argument("--lr", type=float, default=1e-6)
    arg_parser.add_argument("--weight_decay", type=float, default=5e-4)
    arg_parser.add_argument("--momentum", type=float, default=0.99)

    args, _ = arg_parser.parse_known_args()

    summary_writer = SummaryWriter(args.d_summary_dir)

    # 数据集
    train_set = VOC(args.data_dir, True)
    val_set = VOC(args.data_dir, False)

    # 网络和优化器
    network = FCN8s(n_class=train_set.n_classes)
    vgg16 = torchvision.models.vgg16(pretrained=True)
    network.copy_params_from_vgg16(vgg16)
    if args.use_cuda:
        network = network.cuda()

    optimizer = torch.optim.SGD(network.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)

    loss = torch.nn.NLLLoss2d()

    trainer = Trainer(train_set,
                      val_set,
                      args.batch_size,
                      network, optimizer,
                      loss,
                      args.num_epoch,
                      summary_writer,
                      args.use_cuda)
    trainer.run()







