# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import sys
import time
import json
import random

import numpy as np
import tensorflow as tf

from core.Config import *
from core.MixMatch import *
from core.WideResNet import *

from utils.Utils import *
from utils.Dataflow import *
from utils.Generator import *
from utils.Tensorflow_Utils import *

from utils.CIFAR10 import *

if __name__ == '__main__':
    #######################################################################################
    # 1. Config
    #######################################################################################
    flags = get_config()

    num_gpu = len(flags.use_gpu.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.use_gpu
    
    flags.batch_size = flags.batch_size_per_gpu * num_gpu

    model_name = 'MixMatch_cifar@{}'.format(flags.labels)
    model_dir = './experiments/model/{}/'.format(model_name)
    tensorboard_dir = './experiments/tensorboard/{}'.format(model_name)

    ckpt_format = model_dir + '{}.ckpt'
    log_txt_path = model_dir + 'log.txt'

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if os.path.isfile(log_txt_path):
        open(log_txt_path, 'w').close()

    #######################################################################################
    # 2. Dataset
    #######################################################################################
    labeled_dataset, unlabeled_dataset, valid_dataset, test_dataset = get_dataset_cifar10(flags.labels)

    log_print('# labeled dataset : {}'.format(len(labeled_dataset)), log_txt_path)
    log_print('# unlabeled dataset : {}'.format(len(unlabeled_dataset)), log_txt_path)
    log_print('# valid dataset : {}'.format(len(valid_dataset)), log_txt_path)
    log_print('# test dataset : {}'.format(len(test_dataset)), log_txt_path)

    #######################################################################################
    # 3. Generator & Queue
    #######################################################################################
    augment_func = None
    augmentors = []
    if flags.augment == 'weakly_augment':
        augment_func = Weakly_Augment_func
        augmentors.append(Weakly_Augment())

    labeled_dataflow_option = {
        'augmentors' : augmentors,

        'shuffle' : True,
        'remainder' : False,
    
        'batch_size' : flags.batch_size,

        'num_prefetch_for_dataset' : 2,
        'num_prefetch_for_batch' : 2,

        'number_of_cores' : 1,
    }

    unlabeled_dataflow_option = {
        'shuffle' : True,
        'remainder' : False,

        'K' : flags.K,
        'batch_size' : flags.batch_size,
        
        'num_prefetch_for_dataset' : 2,
        'num_prefetch_for_batch' : 2,

        'number_of_cores' : 1,
        'augment_func' : augment_func,
    }

    # for CIFAR-10
    shape = [32, 32, 3]
    classes = 10

    x_image_var = tf.placeholder(tf.float32, [flags.batch_size] + shape)
    x_label_var = tf.placeholder(tf.float32, [flags.batch_size, classes])

    u_image_var = tf.placeholder(tf.float32, [flags.batch_size, flags.K] + shape)

    global_step = tf.placeholder(tf.float32)
    
    labeled_generator_func = lambda ds: Generator({
        'dataset' : ds, 
        'placeholders' : [x_image_var, x_label_var], 
        'queue_size' : 5, 
        'batch_size' : flags.batch_size // num_gpu,
    })

    unlabeled_generator_func = lambda ds: Generator({
        'dataset' : ds, 
        'placeholders' : [u_image_var], 
        'queue_size' : 5, 
        'batch_size' : flags.batch_size // num_gpu,
    })

    labeled_dataflow_list = [generate_labeled_dataflow(labeled_dataset, labeled_dataflow_option) for _ in range(num_gpu)]
    unlabeled_dataflow_list = [generate_unlabeled_dataflow(unlabeled_dataset, unlabeled_dataflow_option) for _ in range(num_gpu)]
    
    labeled_generators = [labeled_generator_func(labeled_dataflow_list[i]) for i in range(num_gpu)]
    unlabeled_generators = [unlabeled_generator_func(unlabeled_dataflow_list[i]) for i in range(num_gpu)]

    log_print('[i] generate dataset and generators', log_txt_path)

    image_op, label_op = labeled_generators[0].dequeue()
    unlabeled_image_op = unlabeled_generators[0].dequeue()

    model_args = dict(
        filters = 32,
        scales = int(np.ceil(np.log2(shape[0]))) - 2,
        repeat = 4,
        getter = None,
    )

    mixmatch = MixMatch({
        'classifier_func' : lambda x: WideResNet(x, True, model_args),

        'K' : flags.K,
        'T' : flags.T,
        
        'alpha' : flags.alpha,
        'batch_size' : flags.batch_size,

        'classes' : classes,
        'shape' : shape,
    })

    x_logits_op, x_label_op, u_predictions_op, u_label_op, train_bn_ops = mixmatch(image_op, label_op, unlabeled_image_op)

    print(image_op, label_op)
    print(unlabeled_image_op)
    print(x_logits_op, x_label_op)
    print(u_predictions_op, u_label_op)
    print(len(train_bn_ops))

    #######################################################################################
    # 4. create Session and Saver
    #######################################################################################
    sess = tf.Session()
    coord = tf.train.Coordinator()

    saver = tf.train.Saver()

    #######################################################################################
    # 5. initialize
    #######################################################################################
    sess.run(tf.global_variables_initializer())

    for labeled_generator in labeled_generators:
        labeled_generator.set_session(sess)
        labeled_generator.set_coordinator(coord)
        labeled_generator.start()

        log_print('[i] start labeled generator ({})'.format(labeled_generator), log_txt_path)

    for unlabeled_generator in unlabeled_generators:
        unlabeled_generator.set_session(sess)
        unlabeled_generator.set_coordinator(coord)
        unlabeled_generator.start()

        log_print('[i] start unlabeled generator ({})'.format(unlabeled_generator), log_txt_path)   
    
    while True:
        image, label, unlabeled_image = sess.run([image_op, label_op, unlabeled_image_op])

        print(image.shape)
        print(label.shape)
        print(unlabeled_image.shape)
        input()
    
