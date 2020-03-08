# coding: utf8
from torch.utils.data import DataLoader
import tqdm
import numpy as np

class Evaluator:
    def __init__(self, valSet, network, nClass, cuda):
        self.valLoader = DataLoader(valSet, batch_size=1, shuffle=False)

        self.nClass = nClass
        self.cuda = cuda

    def evaluate(self, network):
        network.eval()

        totalIous = []
        pixelAccs = []

        for batchIdx, (image, target) in tqdm.tqdm(enumerate(self.valLoader),
                                                  total=len(self.valLoader),
                                                  ncols=80, leave=False):
            if self.cuda:
                image = image.cuda()
                target = target.cuda()

            pred = network(image)
            pred = pred.data.cpu().numpy()

            N, _, h, w = pred.shape

            pred = pred\
                .transpose(0, 2, 3, 1)\
                .reshape(-1, self.nClass)\
                .argmax(axis=1).reshape(N, h, w)

            target = target.cpu().numpy().reshape(N, h, w)
            totalIous.append(self.iou(pred, target))
            pixelAccs.append(self.pixelAcc(pred, target))

        totalIous = np.array(totalIous).T  # n_class * val_len
        ious = np.nanmean(totalIous, axis=1)
        meanIou = np.nanmean(ious)
        pixelAcc = np.array(pixelAccs).mean()

        return meanIou, pixelAcc

    def iou(self, pred, target):
        ious = []
        for cls in range(self.nClass):
            pred_inds = pred == cls
            target_inds = target == cls
            intersection = pred_inds[target_inds].sum()
            union = pred_inds.sum() + target_inds.sum() - intersection
            if union == 0:
                ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
            else:
                ious.append(float(intersection) / max(union, 1))
        return ious

    def pixelAcc(self, pred, target):
        correct = (pred == target).sum()
        total = (target == target).sum()
        return correct / total



