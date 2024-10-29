"""
@author: Saurabh Chatterjee
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2

def gaussian_kernel(size, sigma=1):
    size = int(size)  //2
    x, y = np.mgrid[-size:size+1, -size:size+1]     # return coordinate matrices `x` and `y`, BOTH of **SAME DIMENSION** = No. of indexing dimensions, from coordinate **indices** given for each dimension
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal     # 2-D Gaussian | mean, u = 0 taken
    return g


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float64)
    Gx = ndimage.filters.convolve(img, Kx)
    Gy = ndimage.filters.convolve(img, Ky)
    G = np.hypot(Gx, Gy)
    G = G / G.max() * 255
    theta = np.arctan2(Gy, Gx)

    return (Gx, Gy)



img = cv2.imread('images/blocks.jpg', 0)
# img11 = cv2.imread('images/PCB.jpg')
img = np.asarray(img, np.float32)

Gx, Gy = sobel_filters(img)

# 'M' matrix values:
m11 = np.multiply(Gx, Gx)       # Variance wrt edge pixels along X-dir ideally : but all pixels considered in a given window-size with dirrent weights -> Gaussian
m22 = np.multiply(Gy, Gy)       # Variance wrt edge pixels along Y-dir ideally : but all pixels considered in a given window-size with dirrent weights -> Gaussian
m12 = m21 = np.multiply(Gx, Gy)

g = gaussian_kernel(5, 1)

# Applying Gaussian to each element of 'M' matrix: to make it consider surrounding pixel's Gradients also wrt each pixel to calculate VARIANCES:

m11 = ndimage.convolve(m11, g)
m22 = ndimage.convolve(m22, g)
m12 = ndimage.convolve(m12, g)
m21 = ndimage.convolve(m21, g)


# # x-dir mask:
# x_mask = np.array([[0, 0, 0], [1, 2, 1], [0, 0, 0]], np.float64)

# # y-dir mask:
# y_mask = np.array([[0, 1, 0], [0, 2, 0], [0, 1, 0]], np.float64)

# # x-y dir mask:
# xy_mask = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]], np.float64)

# m11 = ndimage.convolve(m11, x_mask)
# m22 = ndimage.convolve(m22, y_mask)
# m12 = ndimage.convolve(m12, xy_mask)
# m21 = ndimage.convolve(m21, xy_mask)



# Method 1: Determine Eigen Values E1 and E2 => then classify into Corner, Edge or Flat

# Method 2: Calculate "Cornerness-Score" using just Determinant & Trace of 'M' matrix:

trace = m11 + m22

d1 = np.multiply(m11, m22)
d2 = np.multiply(m12, m21)
det = d1 - d2

print(det)

k = 0.11    # 0.11
r = det - k * (np.multiply(trace, trace))   # **Cornerness-Score**

img = cv2.imread('images/blocks.jpg', 0)
imgRGB = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

imgRGB[r > 710000] = np.array([255, 0, 0])     # r>11000 : Large-value means : Corner

plt.figure()

plt.imshow(imgRGB)
plt.show()