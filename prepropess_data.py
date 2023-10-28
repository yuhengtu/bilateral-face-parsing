#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os.path as osp
import os
import cv2
from transform import *
from PIL import Image
# face_data：面部图像数据的路径
# face_sep_mask：面部分割掩码数据的路径
# mask_path：生成的掩码图像的保存路径
face_sep_mask = '/mnt/workspace/face-parsing.PyTorch-master/data/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
mask_path = '/mnt/workspace/face-parsing.PyTorch-master/data/CelebAMask-HQ/mask'
counter = 0
total = 0
for i in range(15): 

    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

    # for j in range(1):
    for j in range(i * 2000, (i + 1) * 2000):

        mask = np.zeros((512, 512))

        for l, att in enumerate(atts, 1):
            total += 1
            file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
            path = osp.join(face_sep_mask, str(i), file_name)
            print(path)

            if os.path.exists(path):
                counter += 1
                sep_mask = np.array(Image.open(path).convert('P'))
                # print(np.unique(sep_mask))

                mask[sep_mask == 225] = l
        # print('{}{}.png'.format(mask_path, j))
        cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)
        print(j)

print(counter, total)