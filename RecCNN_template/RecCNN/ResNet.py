#coding:utf-8
#author: Ziyuan Li, Zhu jiang, Yue Huang, Zhen Qian

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


"""此文件包含了建立 resnet 所用到的类"""

class BasicBlock(layers.Layer):

    """用于建立 resblock"""

    def __init__(self, filter_num, stride=1, need_conv=False):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num[0], (1, 1), strides=stride, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation("relu")

        self.conv2 = layers.Conv2D(filter_num[1], (3, 3), strides=1, padding="same")
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation("relu")

        self.conv3 = layers.Conv2D(filter_num[2], (1, 1), strides=1, padding="same")
        self.bn3 = layers.BatchNormalization()

        if stride != 1 or need_conv:
            self.downsample = Sequential([layers.Conv2D(filter_num[2], (1, 1), strides=stride, padding="same"),
                                          layers.BatchNormalization(axis=3)])
        else:
            self.downsample = lambda x:x

    def call(self, inputs, training=None):

        out = self.conv1(inputs)
        out = self.relu1(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(inputs)
        output = layers.add([out, identity])

        output = tf.nn.relu(output)

        return output


class ResNet(keras.Model):

    """用于建立resnet"""

    def __init__(self, layer_dims, num_classes=3):
        """resnet50: layer_dims [3, 4, 6, 3]，layer_dims为对应resblock中的basicbolk数目"""

        super(ResNet, self).__init__()

        self.stem = Sequential([layers.Conv2D(64, (6, 3), strides=(2, 1), padding="same"),
                                layers.Activation('relu'),
                                layers.BatchNormalization(),
                                layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same"),
                                layers.Activation('relu'),
                                layers.BatchNormalization(),
                                layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")])


        self.layer2 = self.build_resblocks([64, 64, 256], layer_dims[0], stride=1, need_conv=True)
        self.avgpool2 = layers.AveragePooling2D(pool_size=(8, 8), strides=(4, 4))
        self.flatten2 = layers.Flatten()
        self.dsa2 = layers.Dense(100)
        self.dsb2 = layers.Dense(num_classes)

        self.layer3 = self.build_resblocks([128, 128, 512], layer_dims[1], stride=2)
        self.avgpool3 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))
        self.flatten3 = layers.Flatten()
        self.dsa3 = layers.Dense(100)
        self.dsb3 = layers.Dense(num_classes)

        self.layer4 = self.build_resblocks([256, 256, 1024], layer_dims[2], stride=2)
        self.avgpool4 = layers.AveragePooling2D(pool_size=(4, 4), strides=(2, 2))
        self.flatten4 = layers.Flatten()
        self.dsa4 = layers.Dense(100)
        self.dsb4 = layers.Dense(num_classes)

        self.layer5 = self.build_resblocks([512, 512, 2048], layer_dims[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()
        self.dsa = layers.Dense(100)
        self.dsb = layers.Dense(num_classes)

    def call(self, inputs, training=None):

        x = self.stem(inputs)

        x = self.layer2(x)
        x2 = self.avgpool2(x)
        x2 = self.flatten2(x2)
        x2 = self.dsa2(x2)
        x2 = self.dsb2(x2)


        x = self.layer3(x)
        x3 = self.avgpool3(x)
        x3 = self.flatten3(x3)
        x3 = self.dsa3(x3)
        x3 = self.dsb3(x3)

        x = self.layer4(x)
        x4 = self.avgpool4(x)
        x4 = self.flatten4(x4)
        x4 = self.dsa4(x4)
        x4 = self.dsb4(x4)

        x = self.layer5(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dsa(x)
        x = self.dsb(x)

        return tf.concat([x, x2, x3, x4], -1)

    def build_resblocks(self, filter_num, block, stride=1, need_conv=False):
        res_blocks = Sequential()

        res_blocks.add(BasicBlock(filter_num, stride=stride, need_conv=need_conv))
        for i in range(1, block):
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks


