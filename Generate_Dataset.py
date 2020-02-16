# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import argparse

import numpy as np
import tensorflow as tf

from utils.Utils import *

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 1. preprocessing parameters
def parse_args():
    parser = argparse.ArgumentParser(description='MixMatch', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--use_gpu', default='0', type=str)
    parser.add_argument('--seed', default=0, type=int)
    
    parser.add_argument('--num_labels', default=250, type=int)
    parser.add_argument('--num_validation', default=5000, type=int)

    return vars(parser.parse_args())

args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args['use_gpu']

num_labels_per_class = args['num_labels'] // 10
num_valid_per_class = args['num_validation'] // 10

train_dic, test_dataset = get_dataset('./cifar10/')

# 2. 
np.random.seed(args['seed'])

labeled_dataset = []
unlabeled_dataset = []
validation_dataset = []

for class_index in range(10):
    label = one_hot(class_index, 10)
    images = np.asarray(train_dic[class_index])

    np.random.shuffle(images)

    labeled_images = images[:num_labels_per_class]
    unlabeled_images = images[num_labels_per_class:-num_valid_per_class]
    validation_images = images[-num_valid_per_class:]

    labeled_dataset.extend([[image, label] for image in labeled_images])
    unlabeled_dataset.extend([image for image in unlabeled_images])
    validation_dataset.extend([[image, label] for image in validation_images])

print(len(labeled_dataset))
print(len(unlabeled_dataset))
print(len(validation_dataset))
print(len(test_dataset))

# 3. write tfrecords from cifar10 dataset
head_tag = 'cifar@{}_seed@{}'.format(args['num_labels'], args['seed'])
dataset_dir = './dataset/SSL/'

if not os.path.isdir(dataset_dir):
    os.makedirs(dataset_dir)

def write_labeled_dataset(dataset, tfrecord_format, image_per_tfrecord = 5000):
    iteration = len(dataset) // image_per_tfrecord
    remain_length = len(dataset) % image_per_tfrecord

    index = 0
    while index < iteration:
        tfrecord_path = dataset_dir + tfrecord_format.format(index + 1)
        print(tfrecord_path, image_per_tfrecord)

        with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
            for image, label in dataset[index * image_per_tfrecord : (index + 1) * image_per_tfrecord]:
                image_raw = image.tostring()
                label_raw = label.tostring()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'image_raw': _bytes_feature(image_raw),
                    'label_raw': _bytes_feature(label_raw),
                }))
                writer.write(example.SerializeToString())

        index += 1

    if remain_length > 0:
        tfrecord_path = dataset_dir + tfrecord_format.format(index + 1)
        print(tfrecord_path, image_per_tfrecord)

        with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
            for image, label in dataset[-remain_length:]:
                image_raw = image.tostring()
                label_raw = label.tostring()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'image_raw': _bytes_feature(image_raw),
                    'label_raw': _bytes_feature(label_raw),
                }))
                writer.write(example.SerializeToString())

def write_unlabeled_dataset(dataset, tfrecord_format, image_per_tfrecord = 5000):
    iteration = len(dataset) // image_per_tfrecord
    remain_length = len(dataset) % image_per_tfrecord

    index = 0
    while index < iteration:
        tfrecord_path = dataset_dir + tfrecord_format.format(index + 1)
        print(tfrecord_path, image_per_tfrecord)

        with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
            for image in dataset[index * image_per_tfrecord : (index + 1) * image_per_tfrecord]:
                image_raw = image.tostring()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'image_raw': _bytes_feature(image_raw),
                }))
                writer.write(example.SerializeToString())

        index += 1
    
    if remain_length > 0:
        tfrecord_path = dataset_dir + tfrecord_format.format(index + 1)
        print(tfrecord_path, image_per_tfrecord)

        with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
            for image in dataset[-remain_length:]:
                image_raw = image.tostring()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'image_raw': _bytes_feature(image_raw),
                }))
                writer.write(example.SerializeToString())

print(len(labeled_dataset))
print(len(unlabeled_dataset))
print(len(validation_dataset))
print(len(test_dataset))

write_labeled_dataset(labeled_dataset, head_tag + '_labeled' + '_{}.tfrecord')
write_unlabeled_dataset(unlabeled_dataset, head_tag + '_unlabeled' + '_{}.tfrecord')
write_labeled_dataset(validation_dataset, head_tag + '_validation' + '_{}.tfrecord')
write_labeled_dataset(test_dataset, head_tag + '_test' + '_{}.tfrecord')

