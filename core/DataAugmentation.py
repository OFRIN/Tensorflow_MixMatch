import cv2
import random

import numpy as np

from tensorpack import imgaug, dataset
from tensorpack.dataflow import AugmentImageComponent, PrefetchData, BatchData, MultiThreadMapData

def pad(x, border = 4):
    pad_x = np.pad(x, [[border, border], [border, border], [0, 0]], mode = 'reflect')
    return pad_x

def RandomPadandCrop(x):
    new_h, new_w = x.shape[:2]
    x = pad(x, new_w // 8)
    
    h, w = x.shape[:2]
    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    x = x[top: top + new_h, left: left + new_w, :]
    return x

def RandomFlip(x):
    if np.random.rand() < 0.5:
        x = x[:, ::-1, :]
    return x

def Weakly_Augment_func(x):
    x = RandomFlip(x)
    x = RandomPadandCrop(x)
    return x

class Weakly_Augment(imgaug.ImageAugmentor):
    def __init__(self):
        self._init(dict())
    
    def _augment(self, image, _):
        image = RandomFlip(image)
        image = RandomPadandCrop(image)
        return image
