# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import cv2
import copy
import time
import random

import multiprocessing as mp

import numpy as np

from tensorpack import imgaug, dataset
from tensorpack.dataflow import AugmentImageComponent, PrefetchData, BatchData, MultiThreadMapData, RNGDataFlow

from core.DataAugmentation import *

class CIFAR10_Labeled_DataFlow(RNGDataFlow):
    def __init__(self, dataset, option):
        self.option = option
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        indexs = np.arange(len(self.dataset))
        if self.option['shuffle']:
            self.rng.shuffle(indexs)
        
        for i in indexs:
            image, label = self.dataset[i]
            yield [image.astype(np.float32), label.astype(np.float32)]

class CIFAR10_Unlabeled_DataFlow(RNGDataFlow):
    def __init__(self, dataset, option):
        self.option = option
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        indexs = np.arange(len(self.dataset))
        if self.option['shuffle']:
            self.rng.shuffle(indexs)
        
        for i in indexs:
            image = self.dataset[i]
            images = [self.option['augment_func'](image.copy()) for _ in range(self.option['K'])]
            yield [np.asarray(images, np.float32)]

def generate_labeled_dataflow(dataset, option):
    ds = CIFAR10_Labeled_DataFlow(dataset, option)
    ds = AugmentImageComponent(ds, option['augmentors'], copy = False)
    ds = PrefetchData(ds, option['num_prefetch_for_dataset'], option['number_of_cores'])
    
    ds = BatchData(ds, option['batch_size'], remainder = option['remainder'])
    ds = PrefetchData(ds, option['num_prefetch_for_batch'], 2)
    
    return ds

def generate_unlabeled_dataflow(dataset, option):
    ds = CIFAR10_Unlabeled_DataFlow(dataset, option)
    ds = PrefetchData(ds, option['num_prefetch_for_dataset'], option['number_of_cores'])

    ds = BatchData(ds, option['batch_size'], remainder = option['remainder'])
    ds = PrefetchData(ds, option['num_prefetch_for_batch'], 2)
    
    return ds
