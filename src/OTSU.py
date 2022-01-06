# -*- coding:UTF-8 -*-
"""
@author: dueToLife
@contact: wkp372874136@mail.ustc.edu.cn
@datetime: 2021/12/26 21:54
@file: OTSU.py
@software: PyCharm
"""
import cv2
import numpy as np


def otus_match(img1, img2):
    img1gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    [m1,n1]=img1gray.shape
    [m,n]=img2gray.shape
# 阈值分割

    ret1,im1=cv2.threshold(img1gray, 0 ,255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret2,im2=cv2.threshold(img2gray, 0 ,255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.namedWindow('1',cv2.WINDOW_NORMAL)
# cv2.imshow('1', im1)
# cv2.waitKey()
# cv2.namedWindow('2',cv2.WINDOW_NORMAL)
# cv2.imshow('2', im2)
# cv2.waitKey()

    maxn=0
    tempi=0
    tempj=0
    for i in range(m-m1):
        for j in range(n-n1):
            s=np.sum(im1==im2[i:i+m1,j:j+n1])
            if s>maxn:
                maxn=s
                tempi=i
                tempj=j

    img2[tempi:tempi+m1,tempj:tempj+n1,:]=img1
    return img2


if __name__ == '__main__':
    for i in range(28):
        print('processing ', i)
        num = i+1
        num = '{:02d}'.format(num)
        img1 = cv2.imread('../figs/' + str(num) + '_then_cropped.jpg')  # query
        img2 = cv2.imread('../figs/' + str(num) + '_now.jpg')  # train
        res = otus_match(img1, img2)
        cv2.imwrite('../results/' + str(num) + '_otsu.jpg', res)