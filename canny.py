# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 13:48:49 2016

@author: L0472644
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy import misc
from skimage import feature
from PIL import Image
from skimage import measure
import numpy as np
import scipy
from scipy import ndimage

## Ouverture et conversion avec Image de PIL

jpegfile = Image.open('4.jpg')
#jpegfile = jpegfile.resize((108,109),Image.ANTIALIAS)
jpegfile = jpegfile.convert('L')
data = np.array(jpegfile)

## SOBEL

data = ndi.gaussian_filter(data, 4)
edge_horizont = ndimage.sobel(jpegfile, 0,mode='constant')
edge_vertical = ndimage.sobel(jpegfile, 1,mode='constant')
magnitude = np.hypot(edge_horizont, edge_vertical)
magnitude *= 255.0 / np.max(magnitude)  # normalize (Q&D)
angle = np.arctan2(edge_horizont, edge_vertical)

plt.imshow(magnitude)
plt.show()
plt.imshow(angle)
plt.show()


## Ouverture avec misc de scipy et conversion avec rgb2gray de skimage 
from skimage.color import rgb2gray
im1 = misc.imread('1.jpg')
im1 = rgb2gray(im1)
im2 = misc.imread('2.jpg')
im2 = rgb2gray(im2)
im3 = misc.imread('3.jpg')
im3 = rgb2gray(im3)
im4 = misc.imread('4.jpg')
im4 = rgb2gray(im4)


# Canny filter
sigma = 2
low_threshold = .01
high_threshold = .25

edges1 = feature.canny(im1,sigma=sigma,low_threshold=low_threshold, high_threshold=high_threshold)
edges2 = feature.canny(im2, sigma=sigma,low_threshold=low_threshold, high_threshold=high_threshold)
edges3 = feature.canny(im3, sigma=sigma,low_threshold=low_threshold, high_threshold=high_threshold)
edges4 = feature.canny(im4, sigma=sigma,low_threshold=low_threshold, high_threshold=high_threshold)

# Résultats
fig, ax = plt.subplots(nrows=2, ncols=2)

plt.gray()

ax[0,0].imshow(edges1)
ax[0,0].axis('off')


ax[0,1].imshow(edges2)
ax[0,1].axis('off')


ax[1,0].imshow(edges3)
ax[1,0].axis('off')


ax[1,1].imshow(edges4)
ax[1,1].axis('off')

fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)

## Détection des points importants avec le fitre CENSURE de skimage

from skimage.feature import CENSURE

detector = CENSURE()

fig, ax = plt.subplots(nrows=2, ncols=2)

plt.gray()

detector.detect(im1)

ax[0,0].imshow(im1)
ax[0,0].axis('off')
ax[0,0].scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
              2 ** detector.scales, facecolors='none', edgecolors='r')

detector.detect(im2)

ax[0,1].imshow(im2)
ax[0,1].axis('off')
ax[0,1].scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
              2 ** detector.scales, facecolors='none', edgecolors='r')

              
detector.detect(im3)

ax[1,0].imshow(im3)
ax[1,0].axis('off')
ax[1,0].scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
              2 ** detector.scales, facecolors='none', edgecolors='r')


detector.detect(im4)

ax[1,1].imshow(im4)
ax[1,1].axis('off')
ax[1,1].scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
              2 ** detector.scales, facecolors='none', edgecolors='r')

plt.show()

## Filtres de Hough

from skimage.transform import (hough_line, hough_line_peaks)
                               
## Classic Hough (précédé de CANNY)
                               
h1, theta1, d1 = hough_line(edges1)
h2, theta2, d2 = hough_line(edges2)                               
h3, theta3, d3 = hough_line(edges3)                               
h4, theta4, d4 = hough_line(edges4)

fig, ax = plt.subplots(nrows=2, ncols=2)


ax[0,0].imshow(im1, cmap=plt.cm.gray)
rows, cols = im1.shape
for _, angle, dist in zip(*hough_line_peaks(h1, theta1, d1)):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
    ax[0,0].plot((0, cols), (y0, y1), '-r')
ax[0,0].axis((0, cols, rows, 0))
ax[0,0].set_title('Cat1')
ax[0,0].set_axis_off()

ax[0,1].imshow(im2, cmap=plt.cm.gray)
rows, cols = im2.shape
for _, angle, dist in zip(*hough_line_peaks(h2, theta2, d2)):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
    ax[0,1].plot((0, cols), (y0, y1), '-r')
ax[0,1].axis((0, cols, rows, 0))
ax[0,1].set_title('Cat2')
ax[0,1].set_axis_off()

ax[1,0].imshow(im3, cmap=plt.cm.gray)
rows, cols = im3.shape
for _, angle, dist in zip(*hough_line_peaks(h3, theta3, d3)):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
    ax[1,0].plot((0, cols), (y0, y1), '-r')
ax[1,0].axis((0, cols, rows, 0))
ax[1,0].set_title('Cat3')
ax[1,0].set_axis_off()

ax[1,1].imshow(im4, cmap=plt.cm.gray)
rows, cols = im4.shape
for _, angle, dist in zip(*hough_line_peaks(h4, theta4, d4)):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
    ax[1,1].plot((0, cols), (y0, y1), '-r')
ax[1,1].axis((0, cols, rows, 0))
ax[1,1].set_title('Cat4')
ax[1,1].set_axis_off()


# Récupération des y des lignes 
li4 = [0,0,0,0,0,0,0,0,0,0,0,0]
i=0
for _, angle, dist in zip(*hough_line_peaks(h4, theta4, d4)):
    li4[i*2] = (dist - 0 * np.cos(angle)) / np.sin(angle)
    li4[i*2+1] = (dist - cols * np.cos(angle)) / np.sin(angle)
    i=i+1
    
# On fait la diff 2 par 2 
    
fet4 = [0,0,0,0,0,0]
for i in range(0,6):
    fet4[i] = li4[i*2]-li4[i*2+1]