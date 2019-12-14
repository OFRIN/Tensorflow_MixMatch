import sys
sys.path.insert(1, './')

import cv2
import numpy as np

from core.Define import *

train_data_list = np.load('./dataset/train.npy', allow_pickle = True)

image_data_list = []

for train_data in train_data_list:
    image_data_list.append(train_data[0])

image_data_list = np.asarray(image_data_list, dtype = np.float32)

print(np.mean(image_data_list, axis = (0, 1, 2))) # [83.88608 83.88608 83.88608]
print(np.std(image_data_list, axis = (0, 1, 2))) # [68.15831 68.40918 70.49192]

# norm_image = (image_data_list - CIFAR_10_MEAN) / CIFAR_10_STD
# print(norm_image[0].min(), norm_image[0].max())
