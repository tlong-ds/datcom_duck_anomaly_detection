# !/usr/bin/env python

"""
Day 1: Introduction to Image Processing + OpenCV
"""

import cv2
import numpy as np

# Read an image
img = cv2.imread("images.jpeg")

b, g, r = cv2.split(img.astype(np.float32))

gray = 0.299 * r + 0.587 * g + 0.114 * b

gray = np.array(gray.astype(np.uint8))

kernel = (1/9) * np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
])

new_img = np.sum(gray * kernel)
# Show it
cv2.imshow("Original", new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()