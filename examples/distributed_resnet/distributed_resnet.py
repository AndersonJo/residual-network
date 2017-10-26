import argparse
import json
from queue import deque

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from resnet.model import ResNet
from resnet.tool import load_data

parser = argparse.ArgumentParser(description="CIFAR-10 Classification with Deep Residual Neural Network")
parser.add_argument('--mode', default='train', type=str, help='"train" or "test"')
parser.add_argument('--datapath', default='/tmp/cifar10', type=str, help='the directory path to store Iris data set')
parser.add_argument('--epoch', default=30, type=int, )
parser.add_argument('--batch', default=32, type=int, help='batch size')
parser.add_argument('--save_interval', default=5000, type=int,
                    help='Automatically save the model after specific time interval')
parser.add_argument('--visualize_interval', default=20, type=int, help='The interval value to print status like loss')
parser = parser.parse_args()

# Cluster Initialization
cluster_spec = json.load(open('config.json', 'rt'))
cluster = tf.train.ClusterSpec(cluster_spec)
server = tf.train.Server(cluster, job_name='host', task_index=0)


def create_model(resnet: ResNet):
    with tf.variable_scope('input_scope'):
        h = resnet.init_block(filter=[7, 7], channel=[3, 64], max_pool=False)

    with tf.variable_scope('residual01'), tf.device('/job:worker/task:0/gpu:0'):
        h = resnet.residual_block(h, filter=[3, 3], channel=[64, 64])
        h = resnet.residual_block(h, filter=[3, 3], channel=[64, 64])
        h = resnet.residual_block(h, filter=[3, 3], channel=[64, 64])
        h = resnet.residual_block(h, filter=[3, 3], channel=[64, 64])
        h = resnet.residual_block(h, filter=[3, 3], channel=[64, 64])
        h = resnet.residual_block(h, filter=[3, 3], channel=[64, 64])

    with tf.variable_scope('residual02'), tf.device('/job:worker/task:0/gpu:0'):
        h = resnet.max_pool(h, kernel=[2, 2], stride=[2, 2])
        h = resnet.residual_block(h, filter=[3, 3], channel=[64, 128])
        h = resnet.residual_block(h, filter=[3, 3], channel=[128, 128])
        h = resnet.residual_block(h, filter=[3, 3], channel=[128, 128])
        h = resnet.residual_block(h, filter=[3, 3], channel=[128, 128])
        h = resnet.residual_block(h, filter=[3, 3], channel=[128, 128])
        h = resnet.residual_block(h, filter=[3, 3], channel=[128, 128])
        h = resnet.residual_block(h, filter=[3, 3], channel=[128, 128])
        h = resnet.residual_block(h, filter=[3, 3], channel=[128, 128])

    with tf.variable_scope('residual03'), tf.device('/job:worker/task:0/gpu:0'):
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

    with tf.variable_scope('residual04'), tf.device('/job:host/task:0/gpu:0'):
        h = resnet.max_pool(h, kernel=[2, 2], stride=[2, 2])
        h = resnet.residual_block(h, filter=[3, 3], channel=[256, 512])
        h = resnet.residual_block(h, filter=[3, 3], channel=[512, 512])
        h = resnet.residual_block(h, filter=[3, 3], channel=[512, 512])
        h = resnet.residual_block(h, filter=[3, 3], channel=[512, 512])
        h = resnet.residual_block(h, filter=[3, 3], channel=[512, 512])
        h = resnet.residual_block(h, filter=[3, 3], channel=[512, 512])

    with tf.variable_scope('fc'), tf.device('/job:host/task:0/gpu:0'):
        h = resnet.avg_pool(h, kernel=[2, 2], stride=[2, 2])
        h = resnet.fc(h)
    return h


def train(resnet, interval=parser.visualize_interval):
    with tf.variable_scope('fc'), tf.device('/job:host/task:0/gpu:0'):
        loss = resnet.loss()
        adam = tf.train.AdamOptimizer()
        train_op = adam.minimize(loss)
        resnet.compile(target=server.target)

    # Get Data
    train_x, train_y, test_x, test_y = load_data(parser.datapath)

    # Image Augmentation
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, featurewise_center=True,
                                 featurewise_std_normalization=True)
    datagen.fit(train_x)

    iter_count = 0
    _losses = deque(maxlen=interval)
    for epoch in range(1, parser.epoch + 1):
        sample_count = 0

        for i, (sample_x, sample_y) in enumerate(datagen.flow(train_x, train_y, batch_size=resnet.batch)):
            feed_dict = {resnet.x_ts: sample_x, resnet.y_ts: sample_y}
            _loss, _ = resnet.sess.run([loss, train_op], feed_dict=feed_dict)

            # Visualize
            _losses.append(_loss)
            iter_count += 1
            sample_count += 1
            if i % interval == 0:
                _loss = np.mean(_losses)
                print(f'[epoch:{epoch:02}] loss:{_loss:<7.4}  '
                      f'sample_count:{sample_count:<5}  '
                      f'iteration:{iter_count:<5}  ')

            # Save
            if iter_count % parser.save_interval == 0:
                print(f'Model has been successfully saved at iteration = {iter_count}')
                resnet.save()


def evaluate(resnet, batch_size=parser.batch):
    correct_prediction = tf.equal(tf.argmax(resnet.last_layer, 1), y=resnet.y_ts)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    resnet.compile(target=server.target)

    # Get Data
    train_x, train_y, test_x, test_y = load_data(parser.datapath)

    accuracies = list()
    for i in range(0, 10000, batch_size):
        if i + batch_size < 10000:
            _acc = resnet.sess.run(accuracy, feed_dict={
                resnet.x_ts: test_x[i:i + batch_size],
                resnet.y_ts: test_y[i:i + batch_size]})
            accuracies.append(_acc)

    print('Accuracy', np.mean(accuracies))


def main():
    resnet = ResNet(batch=parser.batch)
    create_model(resnet)

    if parser.mode == 'train':
        print('Start Training')
        train(resnet)
    elif parser.mode == 'test':
        print('Start Evaluation')
        evaluate(resnet)


if __name__ == '__main__':
    main()
