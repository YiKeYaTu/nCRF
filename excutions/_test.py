import os
import torch
import torchvision
import cv2
import numpy as np

from os.path import isdir, splitext, join
from PIL import Image


def test(model, dataloader, epoch, test_list, save_dir):
    model.eval()

    if not isdir(save_dir):
        os.makedirs(save_dir)

    for idx, image in enumerate(dataloader):
        image = image.cuda()
        _, _, H, W = image.shape

        results = model(image)
        result = torch.squeeze(results[-1].detach()).cpu().numpy()
        results_all = torch.zeros((len(results), 1, H, W))

        for i in range(len(results)):
            results_all[i, 0, :, :] = results[i]

        filename = splitext(test_list[idx])[0]

        torchvision.utils.save_image(
            1-results_all, join(save_dir, "%s.jpg" % filename))
        result = Image.fromarray((result * 255).astype(np.uint8))
        result.save(join(save_dir, "%s.png" % filename))

        print("Running test [%d/%d]" % (idx + 1, len(dataloader)))


def multiscale_test(model, dataloader, epoch, test_list, save_dir):
    model.eval()

    if not isdir(save_dir):
        os.makedirs(save_dir)

    scale = [0.5, 1, 1.5]

    for idx, image in enumerate(dataloader):
        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))
        _, H, W = image.shape
        multi_fuse = np.zeros((H, W), np.float32)

        for k in range(0, len(scale)):
            im_ = cv2.resize(
                image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2, 0, 1))
            results = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
            result = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
            multi_fuse += fuse

        multi_fuse = multi_fuse / len(scale)
        
        filename = splitext(test_list[idx])[0]

        result_out = Image.fromarray(((1-multi_fuse) * 255).astype(np.uint8))
        result_out.save(join(save_dir, "%s.jpg" % filename))
        result_out_test = Image.fromarray((multi_fuse * 255).astype(np.uint8))
        result_out_test.save(join(save_dir, "%s.png" % filename))

        print("Running test [%d/%d]" % (idx + 1, len(dataloader)))
