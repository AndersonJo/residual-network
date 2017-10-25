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

    def __init__(self, input: tf.Tensor = None, output: tf.Tensor = None, batch: int = 256, n_label: int = 10):
        """
        :param input: Input Tensor. Use tf.placeholder. If not provided input layer for CIFAR-10 is used
        :param output: Output Tensor. Use tf.placeholder. If not provided output layer for CIFAR-10 is used
        :param batch: Batch Size
        :param n_label: The number of labels for classification
        """

        self.batch = batch
        self.n_label = n_label
        self.x_ts = tf.placeholder('float', [None, 32, 32, 3]) if input is None else input
        self.y_ts = tf.placeholder('int64', [None]) if output is None else output

        self.sess = None
        self._names = dict()
        self.layers = list()
        self.layers.append(self.x_ts)

        self.saver = None

    def create_variable(self, name: str, shape: tuple, dtype=tf.float32,
                        initializer=xavier_initializer(), regularizer: str = None):
        if regularizer is not None:
            regularizer = regularizer.lower()
            regularizer = self.regularizers[regularizer]()

        v = tf.get_variable(self._naming(name), shape=shape, dtype=dtype,
                            initializer=initializer, regularizer=regularizer)
        return v

    def conv(self, input_layer, filter: List[int], channel: List[int],
             stride: int, padding: str = 'SAME') -> Tuple[tf.Tensor, tf.Tensor]:
        """
        :param input_layer: Previous layer or tensor
        :param filter: [filter_height, filter_width]
        :param channel: [in_channels, out_channels]
        :param stride:
        :param padding:
        :return: [conv_layer, filter]
        """

        filter_ts = self.create_variable('filter', shape=(*filter, *channel))
        conv = tf.nn.conv2d(input_layer, filter=filter_ts, strides=[1, stride, stride, 1], padding=padding)
        return conv, filter_ts

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
        h, _filter = self.conv(input_layer, filter=filter, channel=channel, stride=stride, padding='SAME')
        h = self.batch_norm(h, out_channel)
        return h

    def init_block(self, filter: List[int] = (7, 7), channel: List[int] = (3, 16),
                   stride: int = 1, max_pool: bool = True) -> tf.Tensor:
        """
        input -> Conv -> ReLU -> output

        :param filter: [filter_height, filter_width]
        :param channel: [in_channels, out_channels]
        :param stride:
        """
        init_conv, _filter = self.conv(self.x_ts, filter=filter, channel=channel, stride=stride)
        init_conv = tf.nn.relu(init_conv)
        if max_pool:
            # MaxPooling
            # ksize: The size of the window for each dimension of the input tensor
            # strides: The stride of the sliding window for each dimension of the input tensor
            output = tf.nn.max_pool(init_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        else:
            output = init_conv

        self.layers.append(output)
        return output

    def max_pool(self, input_layer, kernel: List[int], stride: List[int], padding: str = 'SAME') -> tf.Tensor:
        """
        :param input_layer:
        :param kernel: [width, height] Kernel Size
        :param stride: [width, height] Stirde Size
        :param padding:
        :return:
        """
        k_height, k_width = kernel
        stride_width, stride_height = stride
        output = tf.nn.max_pool(input_layer,
                                ksize=[1, k_height, k_width, 1],
                                strides=[1, stride_width, stride_height, 1], padding=padding)
        self.layers.append(output)
        return output

    def avg_pool(self, input_layer, kernel: List[int], stride: List[int], padding: str = 'SAME') -> tf.Tensor:
        """
        :param input_layer:
        :param kernel: [width, height] Kernel Size
        :param stride: [width, height] Stirde Size
        :param padding:
        :return:
        """
        k_height, k_width = kernel
        stride_width, stride_height = stride
        output = tf.nn.avg_pool(input_layer,
                                ksize=[1, k_height, k_width, 1],
                                strides=[1, stride_width, stride_height, 1], padding=padding)
        self.layers.append(output)
        return output

    def residual_block(self, input_layer, filter: List[int], channel: List[int], stride: int = 1) -> tf.Tensor:
        """
        input -> Conv -> BN -> ReLU -> Conv -> BN -> Addition -> ReLU -> output

        :param input_layer: Usually previous layer
        :param filter: (width<int>, height<int>) The size of the filter
        :param channel: [in_channels, out_channels]
        :param stride: The size of the s
        :return:
        """
        input_channel, output_channel = channel

        h = self.conv_bn(input_layer, filter=filter, channel=[input_channel, output_channel], stride=stride)
        h = tf.nn.relu(h)
        h = self.conv_bn(h, filter=filter, channel=[output_channel, output_channel], stride=stride)

        if input_channel != output_channel:
            # Input channel 과 output channel이 dimension이 다르기 때문에 projection 을 통해서 dimension을 맞춰준다.
            inp, _filter = self.conv(input_layer, filter=[1, 1], channel=[input_channel, output_channel], stride=stride)
        else:
            inp = input_layer

        h = tf.add(h, inp)
        h = tf.nn.relu(h)
        self.layers.append(h)
        return h

    def fc(self, input_layer):
        global_pool = tf.reduce_mean(input_layer, axis=[1, 2])
        fc_w = self.create_variable(name='fc_w', shape=[global_pool.shape[-1], self.n_label])
        fc_b = self.create_variable(name='fc_b', shape=[self.n_label])

        output = tf.matmul(global_pool, fc_w) + fc_b
        self.layers.append(output)
        return output

    def loss(self):
        loss_f = tf.nn.sparse_softmax_cross_entropy_with_logits
        cross_entropy = loss_f(logits=self.last_layer, labels=self.y_ts, name='cross_entropy')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
        return cross_entropy_mean

    def compile(self) -> tf.Session:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())
        self.sess = sess
        return sess

    def save(self, path='/tmp/resnet_anderson.ckpt'):
        if self.saver is None:
            self.saver = tf.train.Saver()
        self.saver.save(self.sess, path)

    def restore(self, path='/tmp/resnet_anderson.ckpt'):
        self.saver.restore(self.sess, path)

    @property
    def last_layer(self) -> tf.Tensor:
        return self.layers[-1]

    def _naming(self, name=None):
        if name is None or not name:
            name = 'variable'
        name = name.lower()
        self._names.setdefault(name, 0)
        self._names[name] += 1
        count = self._names[name]
        return f'{name}_{count:02}'
