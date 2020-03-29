# coding: utf8
"""
模型训练和评估代码
"""
import os
os.system("rm __init__.py")
os.system("rm __init__.pyc")
print(os.listdir("."))

import sys
sys.path.append(".")

import logging
logging.basicConfig(level=logging.INFO)

from src.fcn.fcns import FCN8s
from src.fcn.dataset.voc import VOC
from src.fcn.metrics import Metrics

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
    def __init__(self, train_set, val_set, batch_size, network, optimizer, loss, num_epoch, summary_writer, use_cuda, result_dir):
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

        self.result_dir = result_dir

    def train_epoch(self, epoch):
        self.network.train()

        for batchIdx, (data, target) in enumerate(self.train_loader):
            if self.use_cuda:
                data = data.cuda()
                target = target.cuda()

            self.optimizer.zero_grad()
            pred = self.network(data)
            pred = F.log_softmax(pred, dim=1)

            loss = self.loss(pred, target)
            loss.backward()
            self.optimizer.step()

            if batchIdx % 100 == 0:
                logging.info("epoch: {}, batch: {}, loss: {}".format(epoch, batchIdx, loss.item()))
                self.summary_writer.add_scalar("Train/Loss", loss.item(), epoch * len(self.train_loader) + batchIdx)
                self.summary_writer.flush()

    def evaluate_epoch(self, epoch):

        self.network.eval()
        metrics = Metrics(self.train_set.n_classes)

        sample_idx = random.randint(0, len(self.val_loader) - 1)

        with torch.no_grad():
            for i_val, (images_val, labels_val) in enumerate(self.val_loader):
                if self.use_cuda:
                    images_val = images_val.cuda()
                    labels_val = labels_val.cuda()

                outputs = self.network(images_val)
                val_loss = self.loss(input=outputs, target=labels_val)

                pred = outputs.data.max(1)[1].cpu().numpy()
                gt = labels_val.data.cpu().numpy()

                metrics.update(gt, pred)

                if i_val == sample_idx:
                    pred_mask = self.val_set.label_to_colormap(pred)
                    gt_mask = self.val_set.label_to_colormap(gt)

                    image = self.val_set.get_ori_image(i_val)
                    self.summary_writer.add_image("image", image, dataformats="HWC")
                    self.summary_writer.add_image("pred_mask", pred_mask, dataformats="HWC")
                    self.summary_writer.add_image("gt_mask", gt_mask, dataformats="HWC")
                    self.summary_writer.flush()

                    Image.fromarray(image).save(os.path.join(self.result_dir, "{}.jpg".format(epoch)))
                    Image.fromarray(pred_mask).save(os.path.join(self.result_dir, "{}_pred.jpg".format(epoch)))
                    Image.fromarray(gt_mask).save(os.path.join(self.result_dir, "{}_gt.jpg".format(epoch)))

        scores, cls_iu = metrics.get_scores()

        logging.info("epoch: {}, Overall_Acc: {}, Mean_Acc: {}, Mean_Iou: {}".format(epoch, scores["Overall_Acc"],
                                                                                     scores["Mean_Acc"],
                                                                                     scores["Mean_IoU"]))

        self.summary_writer.add_scalar("Val/Overall_Acc", scores["Overall_Acc"], epoch)
        self.summary_writer.add_scalar("Val/Mean_Acc", scores["Mean_Acc"], epoch)
        self.summary_writer.add_scalar("Val/Mean_IoU", scores["Mean_IoU"], epoch)
        self.summary_writer.flush()

    def run(self):
        for epoch in range(self.num_epoch):
            self.train_epoch(epoch)
            self.evaluate_epoch(epoch)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--d_train_dir', default="E:\\dataset")
    arg_parser.add_argument('--d_val_dir', default=None)
    arg_parser.add_argument('--d_common_data_dir', default=None)
    arg_parser.add_argument('--d_pre_model_dir', default=None)
    arg_parser.add_argument('--d_model_dir', default=None)
    arg_parser.add_argument('--d_summary_dir', default="./run")
    arg_parser.add_argument('--d_result_dir', default="./result")

    arg_parser.add_argument("--data_dir", default="E:\\dataset\\VOCdevkit\\VOC2012")
    arg_parser.add_argument("--use_cuda", type=bool, default=True)
    arg_parser.add_argument("--num_epoch", type=int, default=20)
    arg_parser.add_argument("--batch_size", type=int, default=1)
    arg_parser.add_argument("--lr", type=float, default=1e-6)
    arg_parser.add_argument("--weight_decay", type=float, default=5e-4)
    arg_parser.add_argument("--momentum", type=float, default=0.99)

    args, _ = arg_parser.parse_known_args()

    summary_writer = SummaryWriter(args.d_summary_dir)

    # 解压hdfs数据
    os.system("unzip {0}/VOCdevkit.zip -d {0} > /dev/null 2>&1".format(args.d_train_dir))

    '''
    print(os.listdir("/train_dir/"))
    print(os.listdir("/train_dir/VOCdevkit"))
    print(os.listdir("/train_dir/VOCdevkit/VOC2012"))
    print(os.listdir("/train_dir/VOCdevkit/VOC2012/JPEGImages")[:10])
    '''

    # 数据集
    train_set = VOC(args.data_dir, True)
    val_set = VOC(args.data_dir, False)

    # 网络和优化器
    network = FCN8s(n_class=train_set.n_classes)
    vgg16 = torchvision.models.vgg16(pretrained=False)
    vgg16.load_state_dict(torch.load(os.path.join(args.d_train_dir, "vgg16-397923af.pth")))
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
                      args.use_cuda,
                      args.d_result_dir)
    trainer.run()







