# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import copy
import time
import random

import numpy as np

from threading import Thread

from core.Define import *
from core.DataAugmentation import *

from utils.Utils import *
from utils.StopWatch import *

class Teacher(Thread):
    
    def __init__(self, labeled_data_list, unlabeled_data_list, main_queue):
        Thread.__init__(self)

        self.train = True
        self.watch = StopWatch()
        self.main_queue = main_queue
        
        self.labeled_data_list = copy.deepcopy(labeled_data_list)
        self.unlabeled_data_list = copy.deepcopy(unlabeled_data_list)

        self.labeled_iteration = len(labeled_data_list) // BATCH_SIZE
        self.unlabeled_iteration = len(unlabeled_data_list) // BATCH_SIZE
        
        self.labeled_index = 0
        self.unlabeled_index = 0
        
        np.random.shuffle(self.labeled_data_list)
        np.random.shuffle(self.unlabeled_data_list)
        
    def run(self):
        while self.train:
            while self.main_queue.full():
                time.sleep(0.1)
                continue
            
            li, ui = self.labeled_index, self.unlabeled_index

            batch_x_data_list = self.labeled_data_list[li * BATCH_SIZE : (li + 1) * BATCH_SIZE]
            batch_u_data_list = self.unlabeled_data_list[ui * BATCH_SIZE : (ui + 1) * BATCH_SIZE]
            
            self.labeled_index += 1
            
            if self.labeled_index >= self.labeled_iteration:
                self.labeled_index = 0
                np.random.shuffle(self.labeled_data_list)

            self.unlabeled_index += 1
            if self.unlabeled_index >= self.unlabeled_iteration:
                self.unlabeled_index = 0
                np.random.shuffle(self.unlabeled_data_list)
            
            batch_x_image_list = []
            batch_x_label_list = []
            batch_u_image_list = []

            for x_data in batch_x_data_list:
                image, label = x_data
                image = DataAugmentation(image)
                
                batch_x_image_list.append(image)
                batch_x_label_list.append(label)
            
            for u_data in batch_u_data_list:
                u1_data = DataAugmentation(u_data.copy())
                u2_data = DataAugmentation(u_data.copy())

                batch_u_image_list.append([u1_data, u2_data])
            
            batch_x_image_list = np.asarray(batch_x_image_list, dtype = np.float32)
            batch_x_label_list = np.asarray(batch_x_label_list, dtype = np.float32)
            batch_u_image_list = np.asarray(batch_u_image_list, dtype = np.float32)

            self.main_queue.put([batch_x_image_list, batch_x_label_list, batch_u_image_list])
