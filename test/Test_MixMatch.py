import sys
sys.path.insert(1, '../')

import cv2

import numpy as np
import tensorflow as tf

from core.MixMatch import *
from core.WideResNet import *
from core.Define import *
from core.DataAugmentation import *

from utils.Utils import *
from utils.Teacher_with_MixMatch import *
from utils.Tensorflow_Utils import *

shape = [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL]

x_var = tf.placeholder(tf.float32, [None] + shape, name = 'image/labeled')
u_var = tf.placeholder(tf.float32, [None, K] + shape)
is_training = tf.placeholder(tf.bool)

model_args = dict(filters = 32)

u_reshape = tf.reshape(tf.transpose(u_var, [1, 0, 2, 3, 4]), [-1] + shape)
u_sh_predictions = guess_function(tf.split(u_reshape, K), WideResNet, model_args)

x_label_var = tf.placeholder(tf.float32, [None, CLASSES], name = 'label/labeled')
u_label_op = tf.stop_gradient(u_sh_predictions, name = 'label/unlabeled')

xu_image_op = tf.concat([x_var] + tf.split(u_reshape, K), axis = 0, name = 'xu_image')
xu_label_op = tf.concat([x_label_var] + [u_label_op] * K, axis = 0, name = 'xu_label')

image_ops, label_ops, image_beta_op, label_beta_op = MixMatch(xu_image_op, xu_label_op, xu_image_op, xu_label_op)

# parse labeled, unlabeled
x_image_op, u_image_ops = image_ops[0], image_ops[1:]
x_label_op, u_label_ops = label_ops[0], tf.concat(label_ops[1:], axis = 0)

labeled_data_list, unlabeled_data_list, test_data_list = get_dataset('../dataset/', 250)

np.random.shuffle(labeled_data_list)
np.random.shuffle(unlabeled_data_list)

class_list = [0 for i in range(10)]

labeled_image_data = []
unlabeled_image_data = []
label_data = []

for (image, label) in labeled_data_list:
    image = DataAugmentation(image)

    labeled_image_data.append(image)
    label_data.append(label)
    if len(labeled_image_data) == 3:
        break

for image in unlabeled_data_list:
    u1_image = DataAugmentation(image)
    u2_image = DataAugmentation(image)

    unlabeled_image_data.append([u1_image, u2_image])
    if len(unlabeled_image_data) == 3:
        break

sess = tf.Session()
sess.run(tf.global_variables_initializer())

labeled_image_data = np.asarray(labeled_image_data, dtype = np.float32)
unlabeled_image_data = np.asarray(unlabeled_image_data, dtype = np.float32)
label_data = np.asarray(label_data, dtype = np.float32)

data = sess.run([x_image_op, x_label_op, image_beta_op, label_beta_op], feed_dict = {
    x_var : labeled_image_data, 
    u_var : unlabeled_image_data, 
    x_label_var : label_data,
    is_training : False})

#######################################################################################
# check1. u_reshape = tf.reshape(tf.transpose(u_var, [1, 0, 2, 3, 4]), [-1] + shape)
#######################################################################################
'''
for image_data in unlabeled_image_data:
    u1_image = image_data[0].astype(np.uint8)
    u2_image = image_data[1].astype(np.uint8)

    u1_image = cv2.resize(u1_image, (112, 112))
    u2_image = cv2.resize(u2_image, (112, 112))

    cv2.imshow('show_1', u1_image)
    cv2.imshow('show_2', u2_image)
    cv2.waitKey(0)

for image in data[0]:
    image = cv2.resize(image.astype(np.uint8), (112, 112))

    cv2.imshow('show_reshape', image)
    cv2.waitKey(0)
'''

#######################################################################################
# check2. guess_function (before softmax vs after softmax)
#######################################################################################
# for p1, p2 in zip(data[1], data[2]):
#     print(p1)
#     print(p2)
#     print()
#######################################################################################

#######################################################################################
# check3. sharpening (error O)
#######################################################################################
# for p1, p2 in zip(data[1], data[2]):
#     print(p1)
#     print(p2)
#     print()
#######################################################################################

#######################################################################################
# check4. MixMatch
#######################################################################################
for image, label in zip(data[0], data[1]):
    image = image.astype(np.uint8)
    image = cv2.resize(image, (112, 112))

    print(label, np.argmax(label))
    print(data[2][:, 0, 0, 0], data[3][:, 0])
    cv2.imshow('show', image)
    cv2.waitKey(0)

