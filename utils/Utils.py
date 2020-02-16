# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import cv2
import glob
import pickle

import numpy as np

def log_print(string, log_path = './log.txt'):
    with open(log_path, 'a+') as f:
        print(string)
        f.write(string + '\n')

def get_data(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def one_hot(label, classes):
    v = np.zeros((classes), dtype = np.float32)
    v[label] = 1.
    return v

def get_dataset(dataset_dir):
    train_dic = {}
    test_dataset = []
    
    #########################################################
    # train 
    #########################################################
    for file_path in glob.glob(dataset_dir + "data_batch_*"):
        data = get_data(file_path)
        data_length = len(data[b'filenames'])
        
        for i in range(data_length):
            label = int(data[b'labels'][i])
            image_data = data[b'data'][i]

            channel_size = 32 * 32        

            r = image_data[:channel_size]
            g = image_data[channel_size : channel_size * 2]
            b = image_data[channel_size * 2 : ]

            r = r.reshape((32, 32)).astype(np.uint8)
            g = g.reshape((32, 32)).astype(np.uint8)
            b = b.reshape((32, 32)).astype(np.uint8)

            image = cv2.merge((b, g, r))

            try:
                train_dic[label].append(image)
            except KeyError:
                train_dic[label] = [image]
    
    #########################################################
    # test 
    #########################################################
    data = get_data(dataset_dir + 'test_batch')
    data_length = len(data[b'filenames'])
    
    for i in range(data_length):
        label = int(data[b'labels'][i])
        image_data = data[b'data'][i]

        channel_size = 32 * 32        

        r = image_data[:channel_size]
        g = image_data[channel_size : channel_size * 2]
        b = image_data[channel_size * 2 : ]

        r = r.reshape((32, 32)).astype(np.uint8)
        g = g.reshape((32, 32)).astype(np.uint8)
        b = b.reshape((32, 32)).astype(np.uint8)

        image = cv2.merge((b, g, r))
        test_dataset.append([image, one_hot(label, 10)])
    
    return train_dic, test_dataset

