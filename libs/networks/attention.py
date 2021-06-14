# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.configs import cfgs


def build_attention(inputs, is_training):
    attention_conv3x3_1 = slim.conv2d(inputs, 256, [3, 3],
                                      trainable=is_training,
                                      weights_initializer=cfgs.INITIALIZER,
                                      activation_fn=tf.nn.relu,
                                      scope='attention_conv/3x3_1')
    attention_conv3x3_2 = slim.conv2d(attention_conv3x3_1, 256, [3, 3],
                                      trainable=is_training,
                                      weights_initializer=cfgs.INITIALIZER,
                                      activation_fn=tf.nn.relu,
                                      scope='attention_conv/3x3_2')
    attention_conv3x3_3 = slim.conv2d(attention_conv3x3_2, 256, [3, 3],
                                      trainable=is_training,
                                      weights_initializer=cfgs.INITIALIZER,
                                      activation_fn=tf.nn.relu,
                                      scope='attention_conv/3x3_3')
    attention_conv3x3_4 = slim.conv2d(attention_conv3x3_3, 256, [3, 3],
                                      trainable=is_training,
                                      weights_initializer=cfgs.INITIALIZER,
                                      activation_fn=tf.nn.relu,
                                      scope='attention_conv/3x3_4')
    attention_conv3x3_5 = slim.conv2d(attention_conv3x3_4, 2, [3, 3],
                                      trainable=is_training,
                                      weights_initializer=cfgs.INITIALIZER,
                                      activation_fn=None,
                                      scope='attention_conv/3x3_5')
    return attention_conv3x3_5


def build_inception(inputs, is_training):
    with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                        stride=1, padding='SAME'):
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(inputs, 384, [1, 1],
                                   trainable=is_training,
                                   weights_initializer=cfgs.INITIALIZER,
                                   activation_fn=tf.nn.relu,
                                   scope='conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(inputs, 192, [1, 1],
                                   trainable=is_training,
                                   weights_initializer=cfgs.INITIALIZER,
                                   activation_fn=tf.nn.relu,
                                   scope='conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 224, [1, 7],
                                   trainable=is_training,
                                   weights_initializer=cfgs.INITIALIZER,
                                   activation_fn=tf.nn.relu,
                                   scope='conv2d_0b_1x7')
            branch_1 = slim.conv2d(branch_1, 256, [7, 1],
                                   trainable=is_training,
                                   weights_initializer=cfgs.INITIALIZER,
                                   activation_fn=tf.nn.relu,
                                   scope='conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(inputs, 192, [1, 1],
                                   trainable=is_training,
                                   weights_initializer=cfgs.INITIALIZER,
                                   activation_fn=tf.nn.relu,
                                   scope='conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 192, [7, 1],
                                   trainable=is_training,
                                   weights_initializer=cfgs.INITIALIZER,
                                   activation_fn=tf.nn.relu,
                                   scope='conv2d_0b_7x1')
            branch_2 = slim.conv2d(branch_2, 224, [1, 7],
                                   trainable=is_training,
                                   weights_initializer=cfgs.INITIALIZER,
                                   activation_fn=tf.nn.relu,
                                   scope='Conv2d_0c_1x7')
            branch_2 = slim.conv2d(branch_2, 224, [7, 1],
                                   trainable=is_training,
                                   weights_initializer=cfgs.INITIALIZER,
                                   activation_fn=tf.nn.relu,
                                   scope='conv2d_0d_7x1')
            branch_2 = slim.conv2d(branch_2, 256, [1, 7],
                                   trainable=is_training,
                                   weights_initializer=cfgs.INITIALIZER,
                                   activation_fn=tf.nn.relu,
                                   scope='conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='avgPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1],
                                   trainable=is_training,
                                   weights_initializer=cfgs.INITIALIZER,
                                   activation_fn=tf.nn.relu,
                                   scope='conv2d_0b_1x1')
        inception_out = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        return inception_out


def build_inception_attention(inputs, is_training):
    """Builds Inception-B block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    inception_out = build_inception(inputs, is_training)

    inception_attention_out = slim.conv2d(inception_out, 2, [3, 3],
                                          trainable=is_training,
                                          weights_initializer=cfgs.INITIALIZER,
                                          activation_fn=None,
                                          scope='inception_attention_out')
    return inception_attention_out


def build_context(inputs, is_training):
    conv3x3 = slim.conv2d(inputs, 512, [3, 3],
                          trainable=is_training,
                          weights_initializer=cfgs.INITIALIZER,
                          activation_fn=None,
                          scope='conv3x3')

    conv3x3_dimred = slim.conv2d(inputs, 256, [3, 3],
                                 trainable=is_training,
                                 weights_initializer=cfgs.INITIALIZER,
                                 activation_fn=tf.nn.relu,
                                 scope='conv3x3_dimred')
    conv3x3_5x5 = slim.conv2d(conv3x3_dimred, 256, [3, 3],
                              trainable=is_training,
                              weights_initializer=cfgs.INITIALIZER,
                              activation_fn=None,
                              scope='conv3x3_5x5')

    conv3x3_7x7_1 = slim.conv2d(conv3x3_dimred, 256, [3, 3],
                                trainable=is_training,
                                weights_initializer=cfgs.INITIALIZER,
                                activation_fn=tf.nn.relu,
                                scope='conv3x3_7x7_1')

    conv3x3_7x7 = slim.conv2d(conv3x3_7x7_1, 256, [3, 3],
                              trainable=is_training,
                              weights_initializer=cfgs.INITIALIZER,
                              activation_fn=None,
                              scope='conv3x3_7x7')

    concat_layer = tf.concat([conv3x3, conv3x3_5x5, conv3x3_7x7], axis=-1)

    outputs = tf.nn.relu(concat_layer)
    return outputs


def squeeze_excitation_layer(input_x, out_dim, ratio, layer_name, is_training):
    with tf.name_scope(layer_name):
        # Global_Average_Pooling
        squeeze = tf.reduce_mean(input_x, [1, 2])

        excitation = slim.fully_connected(inputs=squeeze,
                                          num_outputs=out_dim // ratio,
                                          weights_initializer=cfgs.BBOX_INITIALIZER,
                                          activation_fn=tf.nn.relu,
                                          trainable=is_training,
                                          scope=layer_name+'_fully_connected1')

        excitation = slim.fully_connected(inputs=excitation,
                                          num_outputs=out_dim,
                                          weights_initializer=cfgs.BBOX_INITIALIZER,
                                          activation_fn=tf.nn.sigmoid,
                                          trainable=is_training,
                                          scope=layer_name + '_fully_connected2')

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

        # scale = input_x * excitation

        return excitation


def build_attention(inputs, is_training):
    attention_conv3x3_1 = slim.conv2d(inputs, 256, [3, 3],
                                      trainable=is_training,
                                      weights_initializer=cfgs.INITIALIZER,
                                      activation_fn=tf.nn.relu,
                                      scope='attention_conv/3x3_1')
    attention_conv3x3_2 = slim.conv2d(attention_conv3x3_1, 256, [3, 3],
                                      trainable=is_training,
                                      weights_initializer=cfgs.INITIALIZER,
                                      activation_fn=tf.nn.relu,
                                      scope='attention_conv/3x3_2')
    attention_conv3x3_3 = slim.conv2d(attention_conv3x3_2, 256, [3, 3],
                                      trainable=is_training,
                                      weights_initializer=cfgs.INITIALIZER,
                                      activation_fn=tf.nn.relu,
                                      scope='attention_conv/3x3_3')
    attention_conv3x3_4 = slim.conv2d(attention_conv3x3_3, 256, [3, 3],
                                      trainable=is_training,
                                      weights_initializer=cfgs.INITIALIZER,
                                      activation_fn=tf.nn.relu,
                                      scope='attention_conv/3x3_4')
    attention_conv3x3_5 = slim.conv2d(attention_conv3x3_4, 2, [3, 3],
                                      trainable=is_training,
                                      weights_initializer=cfgs.INITIALIZER,
                                      activation_fn=None,
                                      scope='attention_conv/3x3_5')
    return attention_conv3x3_5
