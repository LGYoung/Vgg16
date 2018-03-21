import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np


class Vgg16:

    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.build(self.imgs)
        self.probs = tf.nn.softmax(self.fc8)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)

    def build(self, rgb):
        # build
        # 1
        self.conv1_1 = self.conv_layer(rgb, [3, 3, 3, 64], [1, 1, 1, 1], 'SAME', 'conv1_1', False)
        self.conv1_2 = self.conv_layer(self.conv1_1, [3, 3, 64, 64], [1, 1, 1, 1], 'SAME', 'conv1_2', False)
        self.pool1 = self.max_pool(self.conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', 'pool1')
        # 2
        self.conv2_1 = self.conv_layer(self.pool1, [3, 3, 64, 128], [1, 1, 1, 1], 'SAME', 'conv2_1', False)
        self.conv2_2 = self.conv_layer(self.conv2_1, [3, 3, 128, 128], [1, 1, 1, 1], 'SAME', 'conv2_2', False)
        self.pool2 = self.max_pool(self.conv2_2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', 'pool2')
        # 3
        self.conv3_1 = self.conv_layer(self.pool2, [3, 3, 128, 256], [1, 1, 1, 1], 'SAME', 'conv3_1', False)
        self.conv3_2 = self.conv_layer(self.conv3_1, [3, 3, 256, 256], [1, 1, 1, 1], 'SAME', 'conv3_2', False)
        self.conv3_3 = self.conv_layer(self.conv3_2, [3, 3, 256, 256], [1, 1, 1, 1], 'SAME', 'conv3_3', False)
        self.pool3 = self.max_pool(self.conv3_3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', 'pool3')
        # 4
        self.conv4_1 = self.conv_layer(self.pool3, [3, 3, 256, 512], [1, 1, 1, 1], 'SAME', 'conv4_1', False)
        self.conv4_2 = self.conv_layer(self.conv4_1, [3, 3, 512, 512], [1, 1, 1, 1], 'SAME', 'conv4_2', False)
        self.conv4_3 = self.conv_layer(self.conv4_2, [3, 3, 512, 512], [1, 1, 1, 1], 'SAME', 'conv4_3', False)
        self.pool4 = self.max_pool(self.conv4_3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', 'pool4')
        # 5
        self.conv5_1 = self.conv_layer(self.pool4, [3, 3, 512, 512], [1, 1, 1, 1], 'SAME', 'conv5_1', False)
        self.conv5_2 = self.conv_layer(self.conv5_1, [3, 3, 512, 512], [1, 1, 1, 1], 'SAME', 'conv5_2', False)
        self.conv5_3 = self.conv_layer(self.conv5_2, [3, 3, 512, 512], [1, 1, 1, 1], 'SAME', 'conv5_3', False)
        self.pool5 = self.max_pool(self.conv5_3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', 'pool5')
        # fc
        self.fc6 = self.fc_layer(self.pool5, 4096, 'fc6', True)
        self.fc7 = self.fc_layer(self.fc6, 4096, 'fc7', True)
        self.fc8 = self.fc_layer(self.fc7, 10, 'fc8', True)

    def conv_layer(self, bottom, ksize, strides, padding, name, trainable):
        with tf.variable_scope(name):
            kernel = tf.get_variable(initializer=xavier_initializer(), shape=ksize, dtype=tf.float32,
                                     name='weights', trainable=trainable)
            conv = tf.nn.conv2d(bottom, kernel, strides=strides, padding=padding)
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[ksize[-1]], dtype=tf.float32,
                                     name='biases', trainable=trainable)
            out = tf.nn.relu(tf.nn.bias_add(conv, biases))
            return out

    def fc_layer(self, bottom, output_number, name, trainable):
        with tf.variable_scope(name):
            dim = int(np.prod(bottom.get_shape()[1:]))
            x = tf.reshape(bottom, [-1, dim])
            weights = tf.get_variable(initializer=xavier_initializer(), shape=[dim, output_number],
                                      name='weights', trainable=trainable)
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[output_number], dtype=tf.float32,
                                     trainable=trainable, name='biases')
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            out = tf.nn.relu(fc)
            return out

    def max_pool(self, bottom, ksize, strides, padding, name):
        return tf.nn.max_pool(bottom, ksize=ksize, strides=strides, padding=padding, name=name)

    def avg_pool(self, bottom, ksize, strides, padding, name):
        return tf.nn.avg_pool(bottom, ksize=ksize, strides=strides, padding=padding, name=name)

    def load_weights(self, skip_layers, weight_file, sess):
        weights = np.load(weight_file, encoding='latin1').item()
        for key in weights:
            if key not in skip_layers:
                with tf.variable_scope(key, reuse=True):
                    # for skey, data in zip(('weights', 'biases'), weights[keys])
                    #     sess.run(tf.get_variable(skey).assign(data))
                    sess.run(tf.get_variable('weights').assign(weights[key][0]))
                    sess.run(tf.get_variable('biases').assign(weights[key][1]))



