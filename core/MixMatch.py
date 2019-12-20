import numpy as np
import tensorflow as tf

from core.Define import *

def sharpen(predictions, T):
    predictions = tf.pow(predictions, 1. / T)
    predictions = predictions / tf.reduce_sum(predictions, axis = 1, keep_dims = True)
    return predictions

def guess_function(u_split, option):
    classifier = option['classifier']
    model_args = option['model_args']
    K = option['K']
    T = option['T']

    u_logits = [classifier(u, True, **model_args)[0] for u in u_split]
    u_logits = tf.concat(u_logits, axis = 0)
    
    u_predictions = tf.reshape(u_logits, [K, BATCH_SIZE, CLASSES])
    u_predictions = tf.reduce_mean(u_predictions, axis = 0)
    u_predictions = tf.nn.softmax(u_predictions, axis = -1)
    
    return sharpen(u_predictions, T)

def MixMatch(x1, p1, x2, p2, option):
    n = option['num_sample']
    alpha = option['mixup_alpha']

    n_samples = tf.shape(x1)[0]
    
    beta = tf.distributions.Beta(alpha, alpha).sample([n_samples])
    beta = tf.maximum(beta, 1. - beta)
    
    indexs = tf.random_shuffle(tf.range(tf.shape(x1)[0]))
    x2 = tf.gather(x2, indexs)
    p2 = tf.gather(p2, indexs)
    
    image_beta = tf.reshape(beta, (n_samples, 1, 1, 1))
    label_beta = tf.reshape(beta, (n_samples, 1))

    mix_x = image_beta * x1 + (1 - image_beta) * x2
    mix_y = label_beta * p1 + (1 - label_beta) * p2

    return tf.split(mix_x, n), tf.split(mix_y, n), image_beta, label_beta


