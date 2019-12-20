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

def MixMatch(x1, p1, option):
    n = option['num_sample']
    alpha = option['mixup_alpha']

    beta = tf.distributions.Beta(alpha, alpha).sample(1)[0]
    beta = tf.maximum(beta, 1. - beta)
    
    indexs = tf.random_shuffle(tf.range(tf.shape(x1)[0]))
    x2 = tf.gather(x1, indexs)
    p2 = tf.gather(p1, indexs)
    
    mix_x = beta * x1 + (1 - beta) * x2
    mix_y = beta * p1 + (1 - beta) * p2

    return tf.split(mix_x, n), tf.split(mix_y, n)


