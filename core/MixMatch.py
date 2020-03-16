import numpy as np
import tensorflow as tf

class MixMatch:
    def __init__(self, option):
        self.classifier_func = option['classifier_func']
        
        self.K = option['K']
        self.T = option['T']

        self.alpha = option['alpha']

        self.batch_size = option['batch_size']
        self.classes = option['classes']

        self.shape = option['shape']
    
    def __call__(self, x_image_var, x_label_var, u_image_var):
        # 1. guess label for unlabeled images
        u_image_ops = tf.reshape(tf.transpose(u_image_var, [1, 0, 2, 3, 4]), [-1] + self.shape)
        u_image_ops = tf.split(u_image_ops, self.K)

        u_label_op = self.guess_function(u_image_ops)
        u_label_op = tf.stop_gradient(u_label_op)

        # 2. concatenate images and labels
        xu_image_op = tf.concat([x_image_var] + u_image_ops, axis = 0, name = 'xu_image')
        xu_label_op = tf.concat([x_label_var] + [u_label_op] * self.K, axis = 0, name = 'xu_label')

        # 3. MixUp
        mix_image_op, mix_label_op = self.MixUp(xu_image_op, xu_label_op, xu_image_op, xu_label_op)

        image_ops = tf.split(mix_image_op, self.K + 1)
        label_ops = tf.split(mix_label_op, self.K + 1)

        # 4. Inference

        # 4.1. interleave images
        image_ops = self.interleave(image_ops, self.batch_size)

        # 4.2. split labeled and unlabeled dataset
        x_image_op, u_image_ops = image_ops[0], image_ops[1:]
        x_label_op, u_label_op = label_ops[0], tf.concat(label_ops[1:], axis = 0)

        # 4.3. inference labeled images and calculate batch norm operations
        before_bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        logits_op = [self.classifier_func(x_image_op)['logits']]
        
        after_bn_ops = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if var not in before_bn_ops]

        # 4.4. inference unlabeled images
        logits_op += [self.classifier_func(u)['logits'] for u in u_image_ops]

        # 4.5. interleave
        logits_op = self.interleave(logits_op, self.batch_size)
        
        # 
        x_logits_op = logits_op[0]
        u_logits_op = tf.concat(logits_op[1:], axis = 0)
        u_predictions_op = tf.nn.softmax(u_logits_op, axis = -1)
        
        return x_logits_op, x_label_op, u_predictions_op, u_label_op, after_bn_ops
        
    def sharpen(self, predictions):
        predictions = tf.pow(predictions, 1. / self.T)
        predictions = predictions / tf.reduce_sum(predictions, axis = 1, keep_dims = True)
        return predictions

        # return tf.pow(predictions, 1 / self.T) / tf.reduce_sum(tf.pow(predictions, 1 / self.T), axis = 1, keepdims = True)
    
    def guess_function(self, u_split):
        u_predictions = tf.concat([self.classifier_func(u)['predictions'] for u in u_split], axis = 0)

        u_predictions = tf.reshape(u_predictions, [self.K, self.batch_size, self.classes])
        u_predictions = tf.reduce_mean(u_predictions, axis = 0)
        
        return self.sharpen(u_predictions)
    
    def MixUp(self, x1, p1, x2, p2):
        beta = tf.distributions.Beta(self.alpha, self.alpha).sample([tf.shape(x1)[0], 1, 1, 1])
        beta = tf.maximum(beta, 1. - beta)
        
        indices = tf.random_shuffle(tf.range(tf.shape(x1)[0]))
        xs = tf.gather(x2, indices)
        ps = tf.gather(p2, indices)
        
        mix_x = beta * x1 + (1. - beta) * xs
        mix_y = beta[:, :, 0, 0] * p1 + (1. - beta[:, :, 0, 0]) * ps
        
        return mix_x, mix_y

    # github
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    # github
    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [tf.concat(v, axis=0) for v in xy]
