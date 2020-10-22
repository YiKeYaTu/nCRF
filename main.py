#!/user/bin/python
# coding=utf-8
import sys
import cv2
import torch
import torchvision
import initialize
import atexit

from excutions._train import train
from excutions._test import test, multiscale_test

from os.path import join, split, isfile
from PIL import Image

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from models.Fusion import Fusion

from functions.cross_entropy_loss_nCRF import cross_entropy_loss_nCRF
from constant import args, TMP_DIR, TEST_LIST_DIR

from utils.save_checkpoint import save_checkpoint
from utils.Averagvalue import Averagvalue
from utils.Logger import Logger
from utils.weights_init import normal_weights_init, pretrained_weights_init

from dataloaders.BSDLoader import BSDLoader


def main():
    # dataset
    train_dataset = BSDLoader(root=args.dataset, split="train")
    test_dataset = BSDLoader(root=args.dataset, split="test")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=8, drop_last=True, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers=8, drop_last=True, shuffle=False)

    with open(TEST_LIST_DIR, 'r') as f:
        test_list = f.readlines()

    test_list = [split(i.rstrip())[1] for i in test_list]

    assert len(test_list) == len(test_loader), "%d vs %d" % (
        len(test_list), len(test_loader))

    # model
    # model = N_CRF()
    model = Fusion()
    model.cuda()

    if args.enable_pretrain:
        model.apply(pretrained_weights_init)
    else:
        model.apply(normal_weights_init)

    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # For the nCRF network
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=args.stepsize, gamma=args.gamma)

    # Log
    log = Logger(join(TMP_DIR, '%s-%d-log.txt' % ('sgd', args.lr)))
    sys.stdout = log

    train_loss = []
    train_loss_detail = []

    test(model, test_loader, epoch=None, test_list=test_list,
            save_dir=join(TMP_DIR, 'init-testing-record-view'))

    for epoch in range(args.start_epoch, args.maxepoch):
        tr_avg_loss, tr_detail_loss = train(
            train_loader, model, optimizer, epoch,
            save_dir=join(TMP_DIR, 'epoch-%d-training-record' % epoch))
        test(model, test_loader, epoch=epoch, test_list=test_list,
             save_dir=join(TMP_DIR, 'epoch-%d-testing-record-view' % epoch))
        multiscale_test(model, test_loader, epoch=epoch, test_list=test_list,
                        save_dir=join(TMP_DIR, 'epoch-%d-testing-record' % epoch))
        # write log
        log.flush()
        # Save checkpoint
        save_file = os.path.join(
            TMP_DIR, 'checkpoint_epoch{}.pth'.format(epoch))
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, filename=save_file)
        scheduler.step()  # will adjust learning rate
        # save train/val loss/accuracy, save every epoch in case of early stop
        train_loss.append(tr_avg_loss)
        train_loss_detail += tr_detail_loss


if __name__ == '__main__':
    main()
