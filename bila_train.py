#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet
from face_dataset import FaceMask
from loss import OhemCELoss
from bilateral_solver_conv import BilateralSolverLocal 
from bila_evaluate import evaluate
from optimizer import Optimizer
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import os.path as osp
import logging
import time
import datetime
import argparse


respth = './res'
if not osp.exists(respth):
    os.makedirs(respth)
logger = logging.getLogger()


def train():
    setup_logger(respth)

    # dataset
    n_classes = 19
    n_img_per_gpu = 16
    n_workers = 8
    cropsize = [448, 448]
    data_root = '/mnt/workspace/face-parsing.PyTorch-master/data/CelebAMask-HQ/'

    ds = FaceMask(data_root, cropsize=cropsize, mode='train')
    dl = DataLoader(ds,
                    batch_size = n_img_per_gpu,
                    shuffle = True,
                    num_workers = n_workers,
                    pin_memory = True,
                    drop_last = True)

    # model
    ignore_idx = -100
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.train()
    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1]//16
    LossP = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss_bila = BilateralSolverLocal()
    
    ## optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = 80000
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    optim = Optimizer(
            model = net,
            lr0 = lr_start,
            momentum = momentum,
            wd = weight_decay,
            warmup_steps = warmup_steps,
            warmup_start_lr = warmup_start_lr,
            max_iter = max_iter,
            power = power)

    ## train loop
    msg_iter = 50
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    
    for it in range(max_iter):
    # for it in range(1):
        try:
            im, lb = next(diter) #input和gt，分别来自CelebA-HQ-img和mask, rgb和灰度，reference和target
        except StopIteration:
            epoch += 1
            diter = iter(dl)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out, out16, out32 = net(im)
        lossp = LossP(out, lb)
        loss2 = Loss2(out16, lb)
        loss3 = Loss3(out32, lb)
        # print(lossp + loss2 + loss3)
        # tensor(9.0681, device='cuda:0', grad_fn=<AddBackward0>)
        loss_bila = Loss_bila(output=out, reference=im, target=lb)

        loss = lossp + loss2 + loss3 + loss_bila
    
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())

        #  print training log message
        if (it+1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join([
                    'it: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]).format(
                    it = it+1,
                    max_it = max_iter,
                    lr = lr,
                    loss = loss_avg,
                    time = t_intv,
                    eta = eta
                )
            logger.info(msg)
            loss_avg = []
            st = ed
        if (it+1) % 5000 == 0:
            state = net.state_dict()
            torch.save(state, './res/cp_bila/{}_iter.pth'.format(it))
            evaluate(dspth='/mnt/workspace/face-parsing.PyTorch-master/data/CelebAMask-HQ/CelebA-HQ-img/test-img', cp='{}_iter.pth'.format(it))

    #  dump the final model
    save_pth = osp.join(respth, 'model_final_diss_bila.pth')
    state = net.state_dict()
    torch.save(state, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))


if __name__ == "__main__":
    train()
