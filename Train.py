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

    np.random.seed(flags.seed)

    num_gpu = len(flags.use_gpu.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.use_gpu
    
    flags.max_iteration = flags.max_epochs * flags.valid_iteration
    flags.rampup_iteration = flags.rampup_length * flags.valid_iteration

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
    augmentors = []
    augment_func = None

    if flags.augment == 'weakly_augment':
        number_of_cores = 1

        augment_func = Weakly_Augment_func
        augmentors.append(Weakly_Augment())

    elif flags.augment == 'randaugment':
        number_of_cores = mp.cpu_count()

        # for randaugment

    labeled_dataflow_option = {
        'augmentors' : augmentors,

        'shuffle' : True,
        'remainder' : False,
    
        'batch_size' : flags.batch_size,

        'num_prefetch_for_dataset' : 2,
        'num_prefetch_for_batch' : 2,

        'number_of_cores' : number_of_cores,
    }

    unlabeled_dataflow_option = {
        'shuffle' : True,
        'remainder' : False,

        'K' : flags.K,
        'batch_size' : flags.batch_size,
        
        'num_prefetch_for_dataset' : 2,
        'num_prefetch_for_batch' : 2,

        'number_of_cores' : number_of_cores,
        'augment_func' : augment_func,
    }

    # for CIFAR-10
    shape = [32, 32, 3]
    classes = 10

    x_image_var = tf.placeholder(tf.float32, [flags.batch_size] + shape, name = 'images')
    x_label_var = tf.placeholder(tf.float32, [flags.batch_size, classes], name = 'labels')

    u_image_var = tf.placeholder(tf.float32, [flags.batch_size, flags.K] + shape, name = 'unlabeled_images')

    global_step = tf.placeholder(tf.float32, name = 'step')
    
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

    #######################################################################################
    # 4. Model
    #######################################################################################
    # base(filters) = 32, 1.5M
    # large(filters) = 135, 26M
    model_args = dict(
        filters = 32,
        scales = int(np.ceil(np.log2(shape[0]))) - 2,
        repeat = 4,
        getter = None,
    )

    image_op, label_op = labeled_generators[0].dequeue()
    unlabeled_image_op = unlabeled_generators[0].dequeue()

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

    # calculate loss about labeled dataset and unlabeled dataset
    loss_x_op = tf.nn.softmax_cross_entropy_with_logits_v2(logits = x_logits_op, labels = x_label_op)
    loss_x_op = tf.reduce_mean(loss_x_op)

    lambda_u_op = flags.lambda_u
    if flags.rampup_length > 0:
        lambda_u_op *= tf.clip_by_value(global_step / flags.rampup_iteration, 0.0, 1.0)
    
    loss_u_op = tf.square(u_predictions_op - u_label_op)
    loss_u_op = lambda_u_op * tf.reduce_mean(loss_u_op)

    flags.weight_decay *= flags.learning_rate
    l2_reg_loss_op = flags.weight_decay * tf.add_n([tf.nn.l2_loss(var) for var in [var for var in get_model_vars('WideResNet')]])

    loss_op = loss_x_op + loss_u_op + l2_reg_loss_op

    # set exponential moving average
    ema = tf.train.ExponentialMovingAverage(decay = flags.ema_decay)
    ema_op = ema.apply(get_model_vars())
    
    model_args['getter'] = get_getter(ema)
    predictions_op = WideResNet(image_op, False, model_args)['predictions']

    #######################################################################################
    # 4. Metrics
    #######################################################################################
    # calculate accuracy about labeled dataset
    correct_op = tf.equal(tf.argmax(predictions_op, axis = -1), tf.argmax(label_op, axis = -1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32)) * 100

    #######################################################################################
    # 5. Optimizer
    #######################################################################################
    learning_rate_var = tf.placeholder(tf.float32)
    with tf.control_dependencies(train_bn_ops):
        train_op = tf.train.AdamOptimizer(learning_rate_var).minimize(loss_op, colocate_gradients_with_ops = True)
        train_op = tf.group(train_op, ema_op)

    # for testing
    test_image_var = tf.placeholder(tf.float32, [flags.batch_size] + shape, name = 'images')
    test_label_var = tf.placeholder(tf.float32, [flags.batch_size, classes], name = 'labels')

    test_predictions_op = WideResNet(test_image_var, False, model_args)['predictions']

    # calculate accuracy about labeled dataset
    test_correct_op = tf.equal(tf.argmax(test_predictions_op, axis = -1), tf.argmax(test_label_var, axis = -1))
    test_accuracy_op = tf.reduce_mean(tf.cast(test_correct_op, tf.float32)) * 100

    #######################################################################################
    # 6. Tensorboard
    #######################################################################################
    train_summary_dic = {
        'Loss/Total_Loss' : loss_op,
        'Loss/Labeled_Loss' : loss_x_op,
        'Loss/Unlabeled_Loss' : loss_u_op,
        'Loss/L2_Regularization_Loss' : l2_reg_loss_op,                                                                                                                                                                                                                                                     

        'Accuracy/Train' : accuracy_op,

        'Params/Lambda_U' : lambda_u_op,
        'Params/Learning_rate' : learning_rate_var,
    }
    train_summary_op = tf.summary.merge([tf.summary.scalar(name, train_summary_dic[name]) for name in train_summary_dic.keys()])

    valid_accuracy_var = tf.placeholder(tf.float32)
    valid_accuracy_op = tf.summary.scalar('Accuracy/Validation', valid_accuracy_var)
    
    train_writer = tf.summary.FileWriter(tensorboard_dir)

    log_print('{}'.format(json.dumps(flags_to_dict(flags), indent='\t')), log_txt_path)

    #######################################################################################
    # 7. create Session and Saver
    #######################################################################################
    sess = tf.Session()
    coord = tf.train.Coordinator()

    saver = tf.train.Saver()

    #######################################################################################
    # 8. initialize
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

    loss_list = []
    x_loss_list = []
    u_loss_list = []
    accuracy_list = []
    train_time = time.time()

    best_valid_accuracy = 0.0
    best_valid_ckpt_path = None

    valid_iteration = len(valid_dataset) // flags.batch_size

    train_ops = [train_op, lambda_u_op, loss_op, loss_x_op, loss_u_op, accuracy_op, train_summary_op]

    for iter in range(1, flags.max_iteration + 1):
        _feed_dict = {
            global_step : iter,
            learning_rate_var : flags.learning_rate
        }
        _, weight_u, loss, x_loss, u_loss, accuracy, summary = sess.run(train_ops, feed_dict = _feed_dict)
        train_writer.add_summary(summary, iter)
        
        loss_list.append(loss)
        x_loss_list.append(x_loss)
        u_loss_list.append(u_loss)
        accuracy_list.append(accuracy)
        
        if iter % flags.log_iteration == 0:
            loss = np.mean(loss_list)
            x_loss = np.mean(x_loss_list)
            u_loss = np.mean(u_loss_list)
            accuracy = np.mean(accuracy_list)
            train_time = int(time.time() - train_time)

            log_print('[i] iter = {}, loss = {:.4f}, weight_u = {:.2f}, x_loss = {:.4f}, u_loss = {:.4f}, accuracy = {:.2f}, train_time = {}sec'.format(iter, loss, weight_u, x_loss, u_loss, accuracy, train_time), log_txt_path)
            
            loss_list = []
            x_loss_list = []
            u_loss_list = []
            accuracy_list = []
            train_time = time.time()
        
        if iter % flags.valid_iteration == 0:
            valid_time = time.time()
            valid_accuracy_list = []
            
            for i in range(valid_iteration):
                batch_data_list = valid_dataset[i * flags.batch_size : (i + 1) * flags.batch_size]

                batch_image_data = np.zeros((flags.batch_size, 32, 32, 3), dtype = np.float32)
                batch_label_data = np.zeros((flags.batch_size, 10), dtype = np.float32)
                
                for i, (image, label) in enumerate(batch_data_list):
                    batch_image_data[i] = image.astype(np.float32)
                    batch_label_data[i] = label.astype(np.float32)
                
                _feed_dict = {
                    test_image_var : batch_image_data,
                    test_label_var : batch_label_data,
                }

                accuracy = sess.run(test_accuracy_op, feed_dict = _feed_dict)
                valid_accuracy_list.append(accuracy)

            valid_time = int(time.time() - valid_time)
            valid_accuracy = np.mean(valid_accuracy_list)

            summary = sess.run(valid_accuracy_op, feed_dict = {valid_accuracy_var : valid_accuracy})
            train_writer.add_summary(summary, iter)

            if best_valid_accuracy <= valid_accuracy:
                best_valid_accuracy = valid_accuracy
                best_valid_ckpt_path = ckpt_format.format(iter)

                saver.save(sess, best_valid_ckpt_path)

            log_print('[i] iter = {}, valid_accuracy = {:.2f}, best_valid_accuracy = {:.2f}, valid_time = {}sec'.format(iter, valid_accuracy, best_valid_accuracy, valid_time), log_txt_path)
    
    saver.restore(sess, best_valid_ckpt_path)

    test_time = time.time()
    test_accuracy_list = []
    
    for i in range(len(test_dataset) // flags.batch_size):
        batch_data_list = valid_dataset[i * flags.batch_size : (i + 1) * flags.batch_size]

        batch_image_data = np.zeros((flags.batch_size, 32, 32, 3), dtype = np.float32)
        batch_label_data = np.zeros((flags.batch_size, 10), dtype = np.float32)
        
        for i, (image, label) in enumerate(batch_data_list):
            batch_image_data[i] = image.astype(np.float32)
            batch_label_data[i] = label.astype(np.float32)
        
        _feed_dict = {
            test_image_var : batch_image_data,
            test_label_var : batch_label_data,
        }

        accuracy = sess.run(test_accuracy_op, feed_dict = _feed_dict)
        test_accuracy_list.append(accuracy)

    test_time = int(time.time() - test_time)
    test_accuracy = np.mean(test_accuracy_list)

    log_print('[i] test_accuracy = {:.2f}, test_time = {}sec'.format(test_accuracy, test_time), log_txt_path)
