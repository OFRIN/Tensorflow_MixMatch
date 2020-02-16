# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import cv2
import json
import argparse

import numpy as np
import tensorflow as tf

from core.MixMatch import *
from core.WideResNet import *

from core.WeaklyAugment import *

from utils.Utils import *
from utils.Teacher import *
from utils.Tensorflow_Utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='MixMatch', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--use_gpu', default='0', type=str)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_labels', default=250, type=int)
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--ema_decay', default=0.999, type=float)
    parser.add_argument('--weight_decay', default=0.02, type=float)
    
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--K', default=2, type=int)
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--lambda_u', default=75, type=float) # general = 100
    
    parser.add_argument('--rampup_iteration', default=1024, type=int)
    parser.add_argument('--train_iteration', default=1024 * 1024, type=int)
    
    return vars(parser.parse_args())

args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args['use_gpu']

model_name = 'MixMatch_cifar@{}_seed@{}'.format(args['labels'], args['seed'])

model_dir = './experiments/model/{}/'.format(model_name)
tensorboard_dir = './experiments/tensorboard/{}'.format(model_name)

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

ckpt_format = model_dir + '{}.ckpt'

log_txt_path = model_dir + 'log.txt'
summary_txt_path = model_dir + 'model_summary.txt'

open(log_txt_path, 'w').close()

# 1. dataset
log_print('# {}'.format(model_name), log_txt_path)
log_print('{}'.format(json.dumps(args, indent='\t')), log_txt_path)

# 1.1 get labeled dataset, unlabeled dataset, validation dataset and test_dataset.
train_labeled_dataset = Teacher_for_labeled_dataset(
    {
        'tfrecord_format' : './dataset/SSL/cifar@{}_seed@{}_labeled_*.tfrecord'.format(args['num_labels'], args['seed']),
        
        'is_training' : True,
        'use_prefetch' : False,

        'batch_size' : args['batch_size'],
        'augment_func' : WeaklyAugment,
    }
)

train_unlabeled_dataset = Teacher_for_unlabeled_dataset(
    {
        'tfrecord_format' : './dataset/SSL/cifar@{}_seed@{}_unlabeled_*.tfrecord'.format(args['num_labels'], args['seed']),
        
        'is_training' : True,
        'use_prefetch' : False,

        'K ' : args['K'],
        'batch_size' : args['batch_size'],
        'augment_func' : WeaklyAugment,
    }
)

valid_dataset = Teacher_for_labeled_dataset(
    {
        'tfrecord_format' : './dataset/SSL/cifar@{}_seed@{}_validation_*.tfrecord'.format(args['num_labels'], args['seed']),
        
        'is_training' : False,
        'use_prefetch' : False,

        'batch_size' : args['batch_size'],
        'augment_func' : None,
    }
)

test_dataset = Teacher_for_labeled_dataset(
    {
        'tfrecord_format' : './dataset/SSL/cifar@{}_seed@{}_test_*.tfrecord'.format(args['num_labels'], args['seed']),
        
        'is_training' : False,
        'use_prefetch' : False,

        'batch_size' : args['batch_size'],
        'augment_func' : None,
    }
)

# 2. model
shape = [32, 32, 3]

x_image_var = tf.placeholder(tf.float32, [None] + shape)
x_label_var = tf.placeholder(tf.float32, [None, 10])

u_image_var = tf.placeholder(tf.float32, [None, args['K']] + shape)

global_step = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)

# base(filters) = 32, 1.5M
# large(filters) = 135, 26M
model_args = dict(filters = 32)

# transpose = [16, 2, 32, 32, 3] -> [2, 16, 32, 32, 3]
# reshape = [2 * 16, 32, 32, 3]
u_image_ops = tf.reshape(tf.transpose(u_image_var, [1, 0, 2, 3, 4]), [-1] + shape)
u_image_ops = tf.split(u_image_ops, args['K'])

# guess_function
u_label_op = guess_function(u_image_ops, {
    'classifier' : WideResNet,
    'model_args' : model_args,
    'K' : args['K'],
    'T' : args['T'],
})

u_label_op = tf.stop_gradient(u_label_op)

# concat images and labels.
xu_image_op = tf.concat([x_image_var] + u_image_ops, axis = 0, name = 'xu_image')
xu_label_op = tf.concat([x_label_var] + [u_label_op] * args['K'], axis = 0, name = 'xu_label')

# mixmatch
image_ops, label_ops = MixMatch(xu_image_op, xu_label_op, {
    'K' : args['K'],
    'alpha' : args['alpha'],
})

# interleave 
image_ops = interleave(image_ops, args['batch_size'])

# parse labeled, unlabeled
x_image_op, u_image_ops = image_ops[0], image_ops[1:]
x_label_op, u_label_op = label_ops[0], tf.concat(label_ops[1:], axis = 0)

prior_bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
logits_op = [WideResNet(x_image_op, True, **model_args)[0]]
train_bn_ops = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if var not in prior_bn_ops]

logits_op += [WideResNet(u, True, **model_args)[0] for u in u_image_ops]

# interleave 
logits_op = interleave(logits_op, args['batch_size'])

mix_x_logits_op = logits_op[0]

mix_u_logits_op = tf.concat(logits_op[1:], axis = 0)
mix_u_predictions_op = tf.nn.softmax(mix_u_logits_op)

# calculate Loss_x, Loss_u
loss_x_op = tf.nn.softmax_cross_entropy_with_logits_v2(logits = mix_x_logits_op, labels = x_label_op)
loss_x_op = tf.reduce_mean(loss_x_op)

lambda_u_op = args['lambda_u'] * tf.clip_by_value(global_step / args['rampup_iteration'], 0.0, 1.0)

loss_u_op = tf.square(mix_u_predictions_op - u_label_op)
loss_u_op = lambda_u_op * tf.reduce_mean(loss_u_op)

# with ema
train_vars = tf.trainable_variables()

ema = tf.train.ExponentialMovingAverage(decay = args['ema_decay'])
ema_op = ema.apply(train_vars)

_, predictions_op = WideResNet(x_image_var, False, getter = get_getter(ema), **model_args)

l2_vars = [var for var in train_vars if 'kernel' in var.name or 'weights' in var.name]
l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in l2_vars]) * (args['weight_decay'] * args['learning_rate'])

loss_op = loss_x_op + loss_u_op + l2_reg_loss_op

correct_op = tf.equal(tf.argmax(predictions_op, axis = -1), tf.argmax(x_label_var, axis = -1))
accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32)) * 100

model_summary(train_vars, summary_txt_path)

# 3. optimizer & tensorboard
learning_rate_var = tf.placeholder(tf.float32)
with tf.control_dependencies(train_bn_ops):
    train_op = tf.train.AdamOptimizer(learning_rate_var).minimize(loss_op, colocate_gradients_with_ops = True)
    train_op = tf.group(train_op, ema_op)

train_summary_dic = {
    'Loss/Total_Loss' : loss_op,

    'Loss/Labeled_Loss' : loss_x_op,
    'Loss/Unlabeled_Loss' : loss_u_op,
    'Loss/L2_Regularization_Loss' : l2_reg_loss_op,                                                                                                                                                                                                                                                     
    
    'Monitors/Lambda_U' : lambda_u_op,
    'Monitors/Learning_rate' : learning_rate_var,

    'Accuracy/Train' : accuracy_op,
}
train_summary_op = tf.summary.merge([tf.summary.scalar(name, train_summary_dic[name]) for name in train_summary_dic.keys()])

valid_summary_dic = {
    'Accuracy/Validation' : tf.placeholder(tf.float32),
}
valid_summary_op = tf.summary.merge([tf.summary.scalar(name, valid_summary_dic[name]) for name in valid_summary_dic.keys()])

test_summary_dic = {
    'Accuracy/Test' : tf.placeholder(tf.float32),
}
test_summary_op = tf.summary.merge([tf.summary.scalar(name, test_summary_dic[name]) for name in test_summary_dic.keys()])

train_writer = tf.summary.FileWriter(tensorboard_dir)

# 4. session & saver
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(max_to_keep = 2)

# 5. train
best_valid_path

train_threads = []
main_queue = Queue(20 * NUM_THREADS)

for i in range(NUM_THREADS):
    log_print('# create thread : {}'.format(i), log_txt_path)

    train_thread = Teacher(labeled_data_list, unlabeled_data_list, main_queue)
    train_thread.start()
    
    train_threads.append(train_thread)

learning_rate = args['learning_rate']
train_ops = [train_op, lambda_u_op, loss_op, loss_x_op, loss_u_op, accuracy_op, train_summary_op]

loss_list = []
x_loss_list = []
u_loss_list = []
accuracy_list = []
train_time = time.time()

for iter in range(1, MAX_ITERATION + 1):

    # get batch data with Thread
    batch_x_image_data, batch_x_label_data, batch_u_image_data = main_queue.get()
    
    # print(batch_x_image_data.shape)
    # print(batch_x_label_data.shape)
    # print(batch_u_image_data.shape)
    
    _feed_dict = {
        x_image_var : batch_x_image_data, 
        x_label_var : batch_x_label_data, 
        u_image_var : batch_u_image_data,
        is_training : True,
        learning_rate_var : learning_rate,
        global_step : iter,
    }
    
    _, weight_u, loss, x_loss, u_loss, accuracy, summary = sess.run(train_ops, feed_dict = _feed_dict)
    train_writer.add_summary(summary, iter)
    
    loss_list.append(loss)
    x_loss_list.append(x_loss)
    u_loss_list.append(u_loss)
    accuracy_list.append(accuracy)
    
    if iter % LOG_ITERATION == 0:
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
                x_image_var : batch_image_data,
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
