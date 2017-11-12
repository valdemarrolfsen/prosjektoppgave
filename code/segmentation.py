import argparse
import cv2

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

img = img_as_float(io.imread(args['image']))
io.imshow(img)

for segs in (1000, 2000, 5000, 10000, 100000):
    segments = slic(img, n_segments=segs, sigma=4)
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(mark_boundaries(img, segments))

plt.show()
