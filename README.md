# residual-network
Deep Residual Network 

You can use this library to make your own ResNet. <br>
It is very customizable and use TensorFlow. 

# Installation

The library requires Python 3.6. <br>
Installation is very simple. You can use PIP.

```
pip3 install resnet
```

If you want to install ResNet from source.. 

```
python3.6 setup.py install
```

# Deep Residual Neural Network (ResNet) Example 

## Run ResNet

The Git codes contains CIFAR-10 image classification example. <br>
All you need to do is very simple. 

```
python3.6 main.py --mode=train
```

## Create Your Own ResNet

You might want to customize or make your own ResNet. <br>
The following code shows you how to make your own ResNet. 

```
resnet = ResNet(batch=32)
with tf.variable_scope('input_scope'):
    h = resnet.init_block(filter=[7, 7], channel=[3, 32], max_pool=False)

with tf.variable_scope('residual01'):
    h = resnet.max_pool(h, kernel=[2, 2], stride=[2, 2])
    h = resnet.residual_block(h, filter=[3, 3], channel=[32, 32])
    h = resnet.residual_block(h, filter=[3, 3], channel=[32, 32])
    h = resnet.residual_block(h, filter=[3, 3], channel=[32, 32])
    h = resnet.residual_block(h, filter=[3, 3], channel=[32, 32])
    h = resnet.residual_block(h, filter=[3, 3], channel=[32, 32])
    h = resnet.residual_block(h, filter=[3, 3], channel=[32, 32])

with tf.variable_scope('residual02'):
    h = resnet.max_pool(h, kernel=[2, 2], stride=[2, 2])
    h = resnet.residual_block(h, filter=[3, 3], channel=[32, 64])
    h = resnet.residual_block(h, filter=[3, 3], channel=[64, 64])
    h = resnet.residual_block(h, filter=[3, 3], channel=[64, 64])
    h = resnet.residual_block(h, filter=[3, 3], channel=[64, 64])
    h = resnet.residual_block(h, filter=[3, 3], channel=[64, 64])
    h = resnet.residual_block(h, filter=[3, 3], channel=[64, 64])
    h = resnet.residual_block(h, filter=[3, 3], channel=[64, 64])
    h = resnet.residual_block(h, filter=[3, 3], channel=[64, 64])

with tf.variable_scope('residual03'):
    h = resnet.max_pool(h, kernel=[2, 2], stride=[2, 2])
    h = resnet.residual_block(h, filter=[3, 3], channel=[64, 128])
    h = resnet.residual_block(h, filter=[3, 3], channel=[128, 128])
    h = resnet.residual_block(h, filter=[3, 3], channel=[128, 128])
    h = resnet.residual_block(h, filter=[3, 3], channel=[128, 128])
    h = resnet.residual_block(h, filter=[3, 3], channel=[128, 128])
    h = resnet.residual_block(h, filter=[3, 3], channel=[128, 128])
    h = resnet.residual_block(h, filter=[3, 3], channel=[128, 128])

with tf.variable_scope('residual04'):
    h = resnet.max_pool(h, kernel=[2, 2], stride=[2, 2])
    h = resnet.residual_block(h, filter=[3, 3], channel=[128, 256])
    h = resnet.residual_block(h, filter=[3, 3], channel=[256, 256])
    h = resnet.residual_block(h, filter=[3, 3], channel=[256, 256])
    h = resnet.residual_block(h, filter=[3, 3], channel=[256, 256])
    h = resnet.residual_block(h, filter=[3, 3], channel=[256, 256])

with tf.variable_scope('residual05'):
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
h # <- Your Network Created
```
