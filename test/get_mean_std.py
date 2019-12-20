import sys
sys.path.insert(1, './')

import cv2
import numpy as np

from core.Define import *
from utils.Utils import *

train_dataset, test_dataset = get_dataset_fully_supervised('./dataset/')

print(len(train_dataset))

image_data_list = [train_data[0] for train_data in train_dataset]
image_data_list = np.asarray(image_data_list, dtype = np.float32)

# without normalize
mean = np.mean(image_data_list, axis = (0, 1, 2))
std = np.std(image_data_list, axis = (0, 1, 2))

print(mean)
print(std)

norm_image = (image_data_list - mean) / std
print(norm_image[0].min(), norm_image[0].max())

'''
[83.88608 83.88608 83.88608]
[68.15831 68.40918 70.49192]
-1.2307535 2.472094
'''

# with normalize
# image_data_list /= 255.

# mean = np.mean(image_data_list, axis = (0, 1, 2))
# std = np.std(image_data_list, axis = (0, 1, 2))

# print(mean)
# print(std)

# norm_image = (image_data_list - mean) / std
# print(norm_image[0].min(), norm_image[0].max())

'''
[0.32768 0.32768 0.32768]
[0.26811677 0.26929596 0.27755317]
-1.222154 2.4674594
'''

