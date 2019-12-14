# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import sys
import time
import random
import argparse

import numpy as np
import tensorflow as tf

from queue import Queue

from core.WideResNet import *
from core.Define import *

from utils.Utils import *
from utils.Tensorflow_Utils import *
from utils.Teacher_with_FSL import *

def parse_args():
    parser = argparse.ArgumentParser(description='MixMatch', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_gpu', dest='use_gpu', help='use gpu', default='0', type=str)
    parser.add_argument('--labels', dest='labels', help='labels', default='all', type=str)
    return parser.parse_args()

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu

learning_rate = INIT_LEARNING_RATE

model_name = 'FSL_cifar@{}'.format(args.labels)

model_dir = './experiments/model/{}/'.format(model_name)
tensorboard_dir = './experiments/tensorboard/{}'.format(model_name)

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

ckpt_format = model_dir + '{}.ckpt'
log_txt_path = model_dir + 'log.txt'
summary_txt_path = model_dir + 'model_summary.txt'

open(log_txt_path, 'w').close()

log_print('# Use GPU : {}'.format(args.use_gpu), log_txt_path)
log_print('# learning rate : {}'.format(learning_rate), log_txt_path)
log_print('# batch size : {}'.format(BATCH_SIZE), log_txt_path)
log_print('# max_iteration : {}'.format(MAX_ITERATION), log_txt_path)

# 1. dataset
if args.labels == 'all':
    train_data_list, test_data_list = get_dataset_fully_supervised('./dataset/')
else:
    train_data_list, _, test_data_list = get_dataset('./dataset/', labels)

log_print('# train dataset : {}'.format(len(train_data_list)), log_txt_path)
log_print('# test dataset : {}'.format(len(test_data_list)), log_txt_path)

test_iteration = len(test_data_list) // BATCH_SIZE

# 2. model
shape = [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL]

x_var = tf.placeholder(tf.float32, [BATCH_SIZE] + shape, name = 'image/labeled')
x_label_var = tf.placeholder(tf.float32, [BATCH_SIZE, CLASSES])
is_training = tf.placeholder(tf.bool)

model_args = dict(filters = 135)

# get logits
logits_op = WideResNet(x_var, is_training, **model_args)[0]

# calculate Loss_x, Loss_u
loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits_op, labels = x_label_var)
loss_op = tf.reduce_mean(loss_op)

# with ema
train_vars = tf.get_collection('trainable_variables', 'WideResNet')

ema = tf.train.ExponentialMovingAverage(decay = EMA_DECAY)
ema_op = ema.apply(train_vars)

predictions_op = WideResNet(x_var, is_training, getter = get_getter(ema), **model_args)[1]

l2_vars = [var for var in train_vars if 'kernel' in var.name or 'weights' in var.name]
l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in l2_vars]) * WEIGHT_DECAY
loss_op += l2_reg_loss_op

correct_op = tf.equal(tf.argmax(predictions_op, axis = -1), tf.argmax(x_label_var, axis = -1))
accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32)) * 100

model_summary(train_vars, summary_txt_path)

# 3. optimizer & tensorboard
learning_rate_var = tf.placeholder(tf.float32)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = tf.train.AdamOptimizer(learning_rate_var).minimize(loss_op, colocate_gradients_with_ops = True)
    train_op = tf.group(train_op, ema_op)

train_summary_dic = {
    'Loss/Total_Loss' : loss_op,
    'Loss/L2_Regularization_Loss' : l2_reg_loss_op,                                                                                                                                                                                                                                                     
    'Accuracy/Train' : accuracy_op,
    'Learning_rate' : learning_rate_var,
}

train_summary_list = []
for name in train_summary_dic.keys():
    value = train_summary_dic[name]
    train_summary_list.append(tf.summary.scalar(name, value))
train_summary_op = tf.summary.merge(train_summary_list)

valid_accuracy_var = tf.placeholder(tf.float32)
valid_accuracy_op = tf.summary.scalar('Accuracy/Validation', valid_accuracy_var)

# 4. train loop
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
train_writer = tf.summary.FileWriter(tensorboard_dir)

# 5. thread
best_valid_accuracy = 0.0

train_threads = []
main_queue = Queue(100 * NUM_THREADS)

for i in range(NUM_THREADS):
    log_print('# create thread : {}'.format(i), log_txt_path)

    train_thread = Teacher(train_data_list, BATCH_SIZE, main_queue)
    train_thread.start()
    
    train_threads.append(train_thread)

train_ops = [train_op, loss_op, accuracy_op, train_summary_op]

loss_list = []
accuracy_list = []
train_time = time.time()

for iter in range(1, MAX_ITERATION + 1):

    # get batch data with Thread
    batch_x_image_list, batch_x_label_list = main_queue.get()

    _feed_dict = {
        x_var : batch_x_image_list, 
        x_label_var : batch_x_label_list, 
        is_training : True,
        learning_rate_var : learning_rate,
    }
    
    _, loss, accuracy, summary = sess.run(train_ops, feed_dict = _feed_dict)
    train_writer.add_summary(summary, iter)
    
    loss_list.append(loss)
    accuracy_list.append(accuracy)
    
    if iter % LOG_ITERATION == 0:
        loss = np.mean(loss_list)
        accuracy = np.mean(accuracy_list)
        train_time = int(time.time() - train_time)

        log_print('[i] iter = {}, loss = {:.4f}, accuracy = {:.2f}, train_time = {}sec'.format(iter, loss, accuracy, train_time), log_txt_path)
        
        loss_list = []
        accuracy_list = []
        train_time = time.time()
    
    if iter % SAVE_ITERATION == 0:
        valid_time = time.time()
        valid_accuracy_list = []
        
        for i in range(test_iteration):
            batch_data_list = test_data_list[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

            batch_image_data = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL), dtype = np.float32)
            batch_label_data = np.zeros((BATCH_SIZE, CLASSES), dtype = np.float32)
            
            for i, (image, label) in enumerate(batch_data_list):
                batch_image_data[i] = image.astype(np.float32)
                batch_label_data[i] = label.astype(np.float32)
            
            _feed_dict = {
                x_var : batch_image_data,
                x_label_var : batch_label_data,
                is_training : False
            }

            accuracy = sess.run(accuracy_op, feed_dict = _feed_dict)
            valid_accuracy_list.append(accuracy)

        valid_time = int(time.time() - valid_time)
        valid_accuracy = np.mean(valid_accuracy_list)

        summary = sess.run(valid_accuracy_op, feed_dict = {valid_accuracy_var : valid_accuracy})
        train_writer.add_summary(summary, iter)

        if best_valid_accuracy <= valid_accuracy:
            best_valid_accuracy = valid_accuracy
            saver.save(sess, ckpt_format.format(iter))            

        log_print('[i] iter = {}, valid_accuracy = {:.2f}, best_valid_accuracy = {:.2f}, valid_time = {}sec'.format(iter, valid_accuracy, best_valid_accuracy, valid_time), log_txt_path)

for th in train_threads:
    th.train = False
    th.join()
