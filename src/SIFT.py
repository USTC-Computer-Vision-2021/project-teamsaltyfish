# -*- coding:UTF-8 -*-
"""
@author: dueToLife
@contact: wkp372874136@mail.ustc.edu.cn
@datetime: 2021/12/26 19:53
@file: SIFT.py
@software: PyCharm
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

num = 7
num = '{:02d}'.format(num)
img1 = cv2.imread('../figs/'+str(num)+'_then_cropped.jpg')  # query
img2 = cv2.imread('../figs/'+str(num)+'_now.jpg')  # train

# convert RGB to GRAY
img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

img1hist = cv2.equalizeHist(img1gray)
img2hist = cv2.equalizeHist(img2gray)

img1gauss = cv2.blur(img1, (3, 3))
img2gauss = cv2.blur(img2, (3, 3))
img1gauss = cv2.blur(img1gauss, (3, 3))
img2gauss = cv2.blur(img2gauss, (3, 3))
# img1gauss = cv2.blur(img1gauss, (3, 3))
# img2gauss = cv2.blur(img2gauss, (3, 3))
img1lap = cv2.Laplacian(img1gray, 0) + img1gray
img2lap = cv2.Laplacian(img2gauss, 0) + img2gauss

# create SIFT and detect key points
surf = cv2.SIFT_create()
kp1, description1 = surf.detectAndCompute(img1gauss, None)
kp2, description2 = surf.detectAndCompute(img2gauss, None)

FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)

flann = cv2.FlannBasedMatcher(indexParams, searchParams)
match = flann.knnMatch(description1, description2, k=2)

# well matched points
good = []
# print(len(match))
for i, (m, n) in enumerate(match):
    if m.distance < 0.75*n.distance:
        good.append(m)
# print(len(good))
# projection matrix
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
ano_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, ano_pts, cv2.RANSAC, 4.0)

# apply projection transform to old picture
warpImg = cv2.warpPerspective(img1, M, (img1.shape[1]+img2.shape[1], img1.shape[0]+img2.shape[0]))

# superimpose
for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        p = warpImg[i][j]
        if all(p):
            img2[i][j] = warpImg[i][j]

# save / show
cv2.imwrite('../results/' + str(num) + '_sift.jpg', img2)
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.imshow('result', img2)
cv2.waitKey()


