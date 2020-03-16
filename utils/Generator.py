# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import sys

import threading
import tensorflow as tf

from tensorpack import DataFlowTerminated

class Generator(threading.Thread):
    def __init__(self, option):
        super().__init__()
        self.daemon = True
        
        self.ds = option['dataset']
        self.placeholders = option['placeholders']

        self.queue = tf.FIFOQueue(
            capacity = option['queue_size'],
            dtypes = [ph.dtype for ph in self.placeholders],
            shapes = [[(element if element is not None else option['batch_size']) for element in ph.get_shape().as_list()] for ph in self.placeholders],
        )
        
        self.enqueue_op = self.queue.enqueue(self.placeholders)
        self.close_op = self.queue.close(cancel_pending_enqueues = True)
        
        self.sess = None
        self.coord = None

    def set_session(self, sess):
        self.sess = sess
    
    def set_coordinator(self, coord):
        self.coord = coord

    def size(self):
        return self.sess.run(self.queue.size())

    def run(self):
        with self.sess.as_default():
            try:
                while not self.coord.should_stop():
                    try:
                        self.ds.reset_state()
                        while True:
                            for data in self.ds.get_data():
                                self.enqueue_op.run(feed_dict = dict(zip(self.placeholders, data)))

                    except (tf.errors.CancelledError, tf.errors.OutOfRangeError, DataFlowTerminated):
                        print('[!] coord exception')
                        sys.exit(-1)

                    except Exception as e:
                        print('[!] Exception = {}'.format(str(e)))
                        sys.exit(-1)
            
            except Exception as e:
                print('[!] Exception = {}'.format(str(e)))
            finally:
                try:
                    self.close_op.run()
                except Exception:
                    pass
    
    def dequeue(self):
        return self.queue.dequeue()