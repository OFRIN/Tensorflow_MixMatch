import cv2
import numpy as np

def pad(x, border = 4):
    pad_x = np.pad(x, [[border, border], [border, border], [0, 0]], mode = 'reflect')
    return pad_x

def RandomPadandCrop(x):
    x = pad(x, 4)
    
    h, w = x.shape[:2]
    new_h, new_w = 32, 32
    
    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    x = x[top: top + new_h, left: left + new_w, :]
    
    return x

def RandomFlip(x):
    if np.random.rand() < 0.5:
        x = x[:, ::-1, :]
    return x.copy()

def WeaklyAugment(x):
    x = RandomFlip(x)
    x = RandomPadandCrop(x)
    return x

