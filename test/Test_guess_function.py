import numpy as np
import tensorflow as tf

data = np.random.randint(-5, 5, 3 * 10).reshape((3, 10))

logits_x = tf.placeholder(tf.float32, [3, 10])

output1 = tf.reduce_mean(tf.nn.softmax(logits_x, axis = -1), axis = 0)
output2 = tf.nn.softmax(tf.reduce_mean(logits_x, axis = 0))

sess = tf.Session()
o1 = sess.run(output1, feed_dict = {logits_x : data})
o2 = sess.run(output2, feed_dict = {logits_x : data})

'''
[[-3 -1 -1 -5 -4 -5  0 -4  1  3]
 [-4 -4  1 -2 -1 -3 -1  2  0 -4]
 [-3 -5 -5 -1  1  1 -5 -4  2 -2]]
[0.00241571 0.00564949 0.08013235 0.01296412 0.07792883 0.06899712
 0.02384436 0.20454164 0.24793307 0.27559325]
[0.00881469 0.00881469 0.04666929 0.01716867 0.06513222 0.02396081
 0.03344    0.03344    0.6716603  0.09089933]
'''
print(data)
print(o1)
print(o2)

