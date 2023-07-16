import os
import tensorflow.compat.v1 as tf

import numpy as np
import inspect

import cv2


class VGG19:
    """
    A trainable version VGG19.
    """

    def __init__(self, vgg19_npy_path=None, trainable=True, dropout=0.5):
        if vgg19_npy_path is None:
            path = inspect.getfile(VGG19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path
            print(vgg19_npy_path)

        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        print("npy file loaded")

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout

        self.build()

        self.sess = tf.Session()    # start session
        # initialize variables
        self.sess.run(tf.global_variables_initializer())

    def build(self, train_mode=None):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """
        # tf.placeholder
        self.input = tf.placeholder(
            tf.float32, [None, 224, 224, 3], name='input')
        self.groundtruth = tf.placeholder(
            tf.float32, [None, 3], name='groundtruth')
        self.lr = tf.placeholder(tf.float32, [], name='learning_rate')

        self.conv1_1 = self.conv_layer(self.input, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(
                self.relu6, self.dropout), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(
                self.relu7, self.dropout), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer(self.relu7, 4096, 3, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.groundtruth, logits=self.fc8))
            self.optimizer = tf.train.AdamOptimizer(
                self.lr).minimize(self.loss)
            var_list = [v for v in tf.trainable_variables() if 'fc8' in v.name]
            self.partial_optimizer = tf.train.AdamOptimizer(
                self.lr).minimize(self.loss, var_list=var_list)

        with tf.variable_scope('accuracy'):
            self.correct_prediction = tf.equal(
                tf.argmax(self.prob, 1), tf.argmax(self.groundtruth, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(self.correct_prediction, tf.float32))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(
                3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal(
            [filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, npy_path):

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = self.sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def test(self, batch_x, batch_y):
        fd = {}
        fd[self.input] = batch_x
        fd[self.groundtruth] = batch_y
        prediction, accuracy = self.sess.run(
            [tf.argmax(self.prob), self.accuracy], fd)
        return prediction, accuracy

    def predict(self, img):
        assert img.shape[-1] == 3
        img = cv2.resize(img, (224, 224))
        batch_x = img.reshape((-1, 224, 224, 3))
        fd = {}
        fd[self.input] = batch_x
        predict = self.sess.run(tf.argmax(self.prob, 1), fd)[0]
        label = ['u', 'd', 'm']
        return label[predict]

    def train(self, batch_x, batch_y, lr=1e-4, fine_tune=True):
        fd = {}
        fd[self.input] = batch_x
        fd[self.groundtruth] = batch_y
        fd[self.lr] = lr

        if fine_tune:
            op = self.optimizer
        else:
            op = self.partial_optimizer

        loss, accuracy, _ = self.sess.run(
            [self.loss, self.accuracy, op], fd)
        return loss, accuracy


# if __name__ == '__main__':
#     # Initialize VGG19 object
#     vgg19 = VGG19(vgg19_npy_path='./../weights/fine_tune_weight.npy')
#     # open a BGR image via opencv-python
#     img = cv2.imread('./../data/test-data/0/0.png')

#     # VGG19.predict() take a BGR image as input, and return a label.
#     # Returned label stands for the location of 馬拉巴栗種子芽的點.
#     # Returned label can be either 'u', 'd', or 'm'
#     # which represent up, down , and medium correspondingly.
#     label_predicted = vgg19.predict(img)

#     print(label_predicted)
