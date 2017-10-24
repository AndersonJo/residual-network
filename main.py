import argparse

from model import ResNet
from tool import load_data
import tensorflow as tf

# Parse Arguments
parser = argparse.ArgumentParser(description="CIFAR-10 Classification with Deep Residual Neural Network")
parser.add_argument('--datapath', default='_dataset', type=str, help='the directory path to store Iris data set')
parser = parser.parse_args()


def main():
    train_x, train_y, test_x, test_y = load_data(parser.datapath)

    resnet = ResNet(34)
    resnet.create_model()
    resnet.compile()


if __name__ == '__main__':
    main()
