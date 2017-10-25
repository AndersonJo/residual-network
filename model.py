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

    def __init__(self, batch=256):
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

    def conv(self, input_layer, filter: List[int], channel: List[int], stride: int, padding: str = 'SAME'):
        filter_ts = self.create_variable('filter', shape=(*filter, *channel))
        conv = tf.nn.conv2d(input_layer, filter=filter_ts, strides=[1, stride, stride, 1], padding=padding)
        return conv

    def batch_norm(self, input_layer, dimension):
        mean, variance = tf.nn.moments(input_layer, [0, 1, 2], keep_dims=False)
        beta = self.create_variable('batch_beta', dimension, dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = self.create_variable('batch_gamma', dimension, dtype=tf.float32,
                                     initializer=tf.constant_initializer(1.0, tf.float32))
        bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, self.EPSILON)
        return bn_layer

    def conv_bn(self, input_layer, filter: List[int], channel: List[int], stride: int):
        """
        ResNet에서는 Convolution 다음에는 항상 Batch Normalization을 넣는다.
           "We adopt batch normalization (BN) right after each convolution and before activation"

        filter: [filter_height, filter_width]
        channel: [in_channels, out_channels]
        """
        out_channel = channel[1]
        h = self.conv(input_layer, filter=filter, channel=channel, stride=stride, padding='SAME')
        h = self.batch_norm(h, out_channel)
        return h

    def init_block(self, channel: List[int] = (3, 16)) -> tf.Tensor:
        """
        input -> Conv -> ReLU -> output

        :param channel: [in_channels, out_channels]
        """
        init_conv = self.conv_bn(self.last_layer, filter=[3, 3], channel=channel, stride=1)
        init_conv = tf.nn.relu(init_conv)
        return init_conv

    def residual_block(self, input_layer, filter: List[int], channel: List[int], stride: int = 1) -> tf.Tensor:
        """
        input -> Conv -> BN -> ReLU -> Conv -> BN -> Addition -> ReLU -> output

        :param input_layer: Usually previous layer
        :param filter: (width<int>, height<int>) The size of the filter
        :param channel: [in_channels, out_channels]
        :param stride<int>: The size of the s
        :return:
        """
        input_channel, output_channel = channel

        h = self.conv_bn(input_layer, filter=filter, channel=[input_channel, output_channel], stride=stride)
        h = tf.nn.relu(h)
        h = self.conv_bn(h, filter=filter, channel=[output_channel, output_channel], stride=stride)

        if input_channel != output_channel:
            # Input channel 과 output channel이 dimension이 다르기 때문에 projection 을 통해서 dimension을 맞춰준다.
            inp = self.conv(input_layer, filter=[1, 1], channel=[input_channel, output_channel], stride=stride)
        else:
            inp = input_layer

        h = tf.add(h, inp)
        h = tf.nn.relu(h)
        return h

    def create_model(self, residual_blocks: List):
        pass


        # with tf.variable_scope('block01'):
        #     self.residual_block(init_conv)

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
