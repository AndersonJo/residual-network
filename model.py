from typing import List, Tuple

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import l1_regularizer, l2_regularizer


class ResNet(object):
    """
    Deep Residual Network for CIFAR-10 image classification
    """
    regularizers = {
        'l1': l1_regularizer,
        'l2': l2_regularizer
    }
    EPSILON = 1e-12

    def __init__(self, depth, batch=256):
        self.depth = depth
        self.batch = batch
        self.x_ts = tf.placeholder('float', [batch, 32, 32, 3])
        self.y_ts = tf.placeholder('float', [batch, 10])

        self.sess = None
        self._names = dict()
        self.layers = list()
        self.layers.append(self.x_ts)

    def create_variable(self, name: str, shape: tuple, dtype=tf.float32,
                        initializer=xavier_initializer(), regularizer: str = None):
        if regularizer is not None:
            regularizer = regularizer.lower()
            regularizer = self.regularizers[regularizer]()

        v = tf.get_variable(self._naming(name), shape=shape, dtype=dtype,
                            initializer=initializer, regularizer=regularizer)
        return v

    def batch_norm(self, input_layer, dimension):
        mean, variance = tf.nn.moments(input_layer, [0, 1, 2], keep_dims=False)
        beta = self.create_variable('batch_beta', dimension, dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = self.create_variable('batch_gamma', dimension, dtype=tf.float32,
                                     initializer=tf.constant_initializer(1.0, tf.float32))
        bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, self.EPSILON)
        return bn_layer

    def conv_bn(self, input_layer, filter: Tuple[int, int, int, int], stride: int):
        """
        ResNet에서는 Convolution 다음에는 항상 Batch Normalization을 넣는다.
        "We adopt batch normalization (BN) right after each convolution and before activation"
        filter: [filter_height, filter_width, in_channels, out_channels]
        """
        out_channel = filter[3]
        filter_ts = self.create_variable('filter', shape=filter)
        conv = tf.nn.conv2d(input_layer, filter=filter_ts, strides=[1, stride, stride, 1], padding='SAME')
        bn = self.batch_norm(conv, out_channel)
        out = tf.nn.relu(bn)
        return out

    def residual_block(self, input_layer, output_layer):
        pass

    def create_model(self):
        with tf.variable_scope('input_scope'):
            conv1 = self.conv_bn(self.last_layer, filter=(3, 3, 3, 64), stride=1)
            self.layers.append(conv1)

        # TODO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def compile(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, allow_growth=True)
        sess: tf.InteractiveSession = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())
        self.sess = sess

    @property
    def last_layer(self):
        return self.layers[-1]

    def _naming(self, name=None):
        if name is None or not name:
            name = 'variable'
        name = name.lower()
        self._names.setdefault(name, 0)
        self._names[name] += 1
        count = self._names[name]
        return f'{name}_{count:02}'
