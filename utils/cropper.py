# -*- coding:UTF-8 -*-
"""
@author: dueToLife
@contact: wkp372874136@mail.ustc.edu.cn
@datetime: 2021/12/26 20:43
@file: cropper.py
@software: PyCharm
"""
import os
import re
import random
import cv2
PATH = '../figs/'
files = os.listdir(PATH)
pattern = re.compile(r'.*then.*')
for file in files:
    if pattern.match(file):
        img = cv2.imread(PATH+file)
        height, width = img.shape[0], img.shape[1]
        crop_height, crop_width = int(height/2) - 1, int(width/2) - 1
        x, y = random.randint(0, crop_height), random.randint(0, crop_width)
        img = img[x:x+crop_height, y:y+crop_width]
        # cv2.imshow('crop_picture', img)
        # cv2.waitKey(0)
        strs = file.split('.')
        cropped_file = strs[0] + '_cropped.' + strs[1]
        cv2.imwrite(PATH+cropped_file, img)
