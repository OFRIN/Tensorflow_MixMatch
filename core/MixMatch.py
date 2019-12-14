import numpy as np
import tensorflow as tf

from core.Define import *

def sharpen(predictions):
    predictions = tf.pow(predictions, 1. / T)
    predictions = predictions / tf.reduce_sum(predictions, axis = 1, keep_dims = True)
    return predictions

def guess_function(u_split, classifier, model_args):
    u_logits = [classifier(u, True, **model_args)[0] for u in u_split]
    u_logits = tf.concat(u_logits, axis = 0)
    
    u_predictions = tf.reshape(u_logits, [K, BATCH_SIZE, CLASSES])
    u_predictions = tf.reduce_mean(u_predictions, axis = 0)
    u_predictions = tf.nn.softmax(u_predictions)
    
    return sharpen(u_predictions)

def MixMatch(x1, p1, x2, p2, beta = BETA, n = K + 1):
    beta = tf.distributions.Beta(beta, beta).sample([tf.shape(x1)[0], 1, 1, 1])
    beta = tf.maximum(beta, 1 - beta)
    
    indexs = tf.random_shuffle(tf.range(tf.shape(x1)[0]))
    x2 = tf.gather(x2, indexs)
    p2 = tf.gather(p2, indexs)
    
    mix_x = beta * x1 + (1 - beta) * x2
    mix_y = beta[:, :, 0, 0] * p1 + (1 - beta[:, :, 0, 0]) * p2
    
    return tf.split(mix_x, n), tf.split(mix_y, n)


