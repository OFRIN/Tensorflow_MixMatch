import sys
sys.path.insert(1, '../')

import cv2
import numpy as np

from core.DataAugmentation import *

from utils.Utils import *
from utils.Teacher_with_MixMatch import *
from utils.Tensorflow_Utils import *

labeled_data_list, unlabeled_data_list, test_data_list = get_dataset('../dataset/', 250)

class_list = [0 for i in range(10)]

np.random.shuffle(labeled_data_list)

for (image, label) in labeled_data_list:
    class_list[np.argmax(label)] += 1
    print(label, np.argmax(label))

    # aug_image = DataAugmentation(image)
    pad_x = np.pad(image, [[4, 4], [4, 4], [0, 0]], mode = 'reflect')
    aug_image = RandomPadandCrop(image).astype(np.uint8)

    image = cv2.resize(image, (112, 112))
    pad_x = cv2.resize(pad_x, (112, 112))
    aug_image = cv2.resize(aug_image, (112, 112))

    cv2.imshow('show', image)
    cv2.imshow('show with pad', pad_x)
    cv2.imshow('show with augment', aug_image)
    cv2.waitKey(0)

print(class_list)
