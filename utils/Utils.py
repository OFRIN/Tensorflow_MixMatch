import cv2
import glob

import numpy as np

def log_print(string, log_path = './log.txt'):
    print(string)
    
    if log_path is not None:
        f = open(log_path, 'a+')
        f.write(string + '\n')
        f.close()

def one_hot(label, classes):
    v = np.zeros((classes), dtype = np.float32)
    v[label] = 1.
    return v
