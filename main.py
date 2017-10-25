
import argparse

from model import ResNet
from tool import load_data
import tensorflow as tf

# Parse Arguments
parser = argparse.ArgumentParser(description="CIFAR-10 Classification with Deep Residual Neural Network")
parser.add_argument('--datapath', default='_dataset', type=str, help='the directory path to store Iris data set')
parser = parser.parse_args()


def create_model():
    resnet = ResNet()
    with tf.variable_scope('input_scope'):
        h = resnet.init_block(filter=[7, 7], channel=[3, 64], max_pool=False)

    with tf.variable_scope('residual01'):
        h = resnet.residual_block(h, filter=[3, 3], channel=[64, 64])
        h = resnet.residual_block(h, filter=[3, 3], channel=[64, 64])
        h = resnet.residual_block(h, filter=[3, 3], channel=[64, 64])
        h = resnet.residual_block(h, filter=[3, 3], channel=[64, 64])
        h = resnet.residual_block(h, filter=[3, 3], channel=[64, 64])
        h = resnet.residual_block(h, filter=[3, 3], channel=[64, 64])

    with tf.variable_scope('residual02'):
        h = resnet.max_pool(h, kernel=[2, 2], stride=[2, 2])
        h = resnet.residual_block(h, filter=[3, 3], channel=[64, 128])
        h = resnet.residual_block(h, filter=[3, 3], channel=[128, 128])
        h = resnet.residual_block(h, filter=[3, 3], channel=[128, 128])
        h = resnet.residual_block(h, filter=[3, 3], channel=[128, 128])
        h = resnet.residual_block(h, filter=[3, 3], channel=[128, 128])
        h = resnet.residual_block(h, filter=[3, 3], channel=[128, 128])
        h = resnet.residual_block(h, filter=[3, 3], channel=[128, 128])
        h = resnet.residual_block(h, filter=[3, 3], channel=[128, 128])

    with tf.variable_scope('residual03'):
        h = resnet.max_pool(h, kernel=[2, 2], stride=[2, 2])
        h = resnet.residual_block(h, filter=[3, 3], channel=[128, 256])
        h = resnet.residual_block(h, filter=[3, 3], channel=[256, 256])
        h = resnet.residual_block(h, filter=[3, 3], channel=[256, 256])
        h = resnet.residual_block(h, filter=[3, 3], channel=[256, 256])
        h = resnet.residual_block(h, filter=[3, 3], channel=[256, 256])
        h = resnet.residual_block(h, filter=[3, 3], channel=[256, 256])
        h = resnet.residual_block(h, filter=[3, 3], channel=[256, 256])
        h = resnet.residual_block(h, filter=[3, 3], channel=[256, 256])
        h = resnet.residual_block(h, filter=[3, 3], channel=[256, 256])
        h = resnet.residual_block(h, filter=[3, 3], channel=[256, 256])
        h = resnet.residual_block(h, filter=[3, 3], channel=[256, 256])
        h = resnet.residual_block(h, filter=[3, 3], channel=[256, 256])

    with tf.variable_scope('residual04'):
        h = resnet.max_pool(h, kernel=[2, 2], stride=[2, 2])
        h = resnet.residual_block(h, filter=[3, 3], channel=[256, 512])
        h = resnet.residual_block(h, filter=[3, 3], channel=[512, 512])
        h = resnet.residual_block(h, filter=[3, 3], channel=[512, 512])
        h = resnet.residual_block(h, filter=[3, 3], channel=[512, 512])
        h = resnet.residual_block(h, filter=[3, 3], channel=[512, 512])
        h = resnet.residual_block(h, filter=[3, 3], channel=[512, 512])

    with tf.variable_scope('fc'):
        h = resnet.avg_pool(h, kernel=[2, 2], stride=[2, 2])
        h = resnet.fc(h)
    return resnet, h


def train():
    pass


def main():
    resnet, last_layer = create_model()
    loss = resnet.loss(last_layer)
    adam = tf.train.AdamOptimizer()
    train_op = adam.minimize(loss)

    train()
    resnet.compile()
    train_x, train_y, test_x, test_y = load_data(parser.datapath)


if __name__ == '__main__':
    main()
