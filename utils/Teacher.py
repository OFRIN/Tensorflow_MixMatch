# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import cv2
import random

import numpy as np
import tensorflow as tf

from utils.Utils import *

class Teacher_for_labeled_dataset:
    def __init__(self, option):
        self.augment_func = option['augment_func']

        dataset = tf.data.Dataset.list_files(option['tfrecord_format'], shuffle = option['is_training'])
        if option['is_training']: 
            dataset = dataset.repeat()

        dataset = dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename, buffer_size = 16 * 1024 * 1024), 
            cycle_length = 16, 
            num_parallel_calls = tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.shuffle(1024)
        
        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(
                self.parser,
                batch_size = option['batch_size'],
                num_parallel_calls = 2,
                drop_remainder = option['is_training'],
            )
        )
        
        if option['use_prefetch']:
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.iterator = dataset.make_initializable_iterator()
        self.image_op, self.label_op = self.iterator.get_next()
        self.initializer_op = self.iterator.initializer

    def parser(self, record):
        parsed = tf.parse_single_example(
            record, 
            features = {
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label_raw': tf.FixedLenFeature([], tf.string),
        })

        image = tf.decode_raw(parsed['image_raw'], tf.uint8)
        image = tf.reshape(image, [32, 32, 3])
        image = tf.cast(image, tf.float32)

        label = tf.decode_raw(parsed['label_raw'], tf.float32)
        label = tf.reshape(label, [10])
        
        if self.augment_func is not None:
            [image] = tf.py_func(self.augment_func, [image], [tf.float32])

        return image, label

class Teacher_for_unlabeled_dataset:
    def __init__(self, option):
        self.K = option['K']
        self.augment_func = option['augment_func']

        dataset = tf.data.Dataset.list_files(option['tfrecord_format'], shuffle = option['is_training'])
        if option['is_training']: 
            dataset = dataset.repeat()

        dataset = dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename, buffer_size = 16 * 1024 * 1024), 
            cycle_length = 16, 
            num_parallel_calls = tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.shuffle(1024)

        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(
                self.parser,
                batch_size = option['batch_size'],
                num_parallel_calls = 2,
                drop_remainder = option['is_training'],
            )
        )

        if option['use_prefetch']:
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.iterator = dataset.make_initializable_iterator()
        self.image_op, self.label_op = self.iterator.get_next()
        self.initializer_op = self.iterator.initializer

    def parser(self, record):
        parsed = tf.parse_single_example(
            record, 
            features = {
                'image_raw': tf.FixedLenFeature([], tf.string),
        })

        image = tf.decode_raw(parsed['image_raw'], tf.uint8)
        image = tf.reshape(image, [32, 32, 3])
        image = tf.cast(image, tf.float32)

        images = []
        for _ in range(self.K):
            images += tf.py_func(self.augment_func, [image], [tf.float32])
        
        return images


