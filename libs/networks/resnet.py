# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division


import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.configs import cfgs
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
from libs.networks.layer import squeeze_excitation_layer, build_attention, build_inception, build_inception_attention
from help_utils.tools import add_heatmap


def resnet_arg_scope(
        is_training=True, weight_decay=cfgs.WEIGHT_DECAY, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    '''

    In Default, we do not use BN to train resnet, since batch_size is too small.
    So is_training is False and trainable is False in the batch_norm params.

    '''
    batch_norm_params = {
        'is_training': False, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


def resnet_base(img_batch, scope_name, is_training=True):
    '''
    this code is derived from light-head rcnn.
    https://github.com/zengarden/light_head_rcnn

    It is convenient to freeze blocks. So we adapt this mode.
    '''
    if scope_name == 'resnet_v1_50':
        middle_num_units = 6
    elif scope_name == 'resnet_v1_101':
        middle_num_units = 23
    else:
        raise NotImplementedError('We only support resnet_v1_50 or resnet_v1_101 or mobilenetv2. '
                                  'Check your network name.')

    blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
              resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
              # use stride 1 for the last conv4 layer.

              resnet_v1_block('block3', base_depth=256, num_units=middle_num_units, stride=1)]
              # when use fpn, stride list is [1, 2, 2]

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        with tf.variable_scope(scope_name, scope_name):
            # Do the first few layers manually, because 'SAME' padding can behave inconsistently
            # for images of different sizes: sometimes 0, sometimes 1
            net = resnet_utils.conv2d_same(
                img_batch, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='VALID', scope='pool1')

    not_freezed = [False] * cfgs.FIXED_BLOCKS + (4-cfgs.FIXED_BLOCKS)*[True]
    # Fixed_Blocks can be 1~3

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[0]))):
        C2, end_points_C2 = resnet_v1.resnet_v1(net,
                                                blocks[0:1],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    # C2 = tf.Print(C2, [tf.shape(C2)], summarize=10, message='C2_shape')

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[1]))):
        C3, end_points_C3 = resnet_v1.resnet_v1(C2,
                                                blocks[1:2],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    # C3 = tf.Print(C3, [tf.shape(C3)], summarize=10, message='C3_shape')

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[2]))):
        C4, _ = resnet_v1.resnet_v1(C3,
                                    blocks[2:3],
                                    global_pool=False,
                                    include_root_block=False,
                                    scope=scope_name)

        if cfgs.ADD_FUSION:

            # C3_ = end_points_C3['{}/block2/unit_3/bottleneck_v1'.format(scope_name)]
            # # channels = C3_.get_shape().as_list()
            # filters1 = tf.random_normal([3, 3, 512, 1024], mean=0.0, stddev=0.01)
            # C3_atrous_conv2d = tf.nn.atrous_conv2d(C3_, filters=filters1, rate=2, padding='SAME')
            # C3_shape = tf.shape(C3_atrous_conv2d)
            #
            # C2_ = end_points_C2['{}/block1/unit_2/bottleneck_v1'.format(scope_name)]
            # filters2 = tf.random_normal([3, 3, 256, 512], mean=0.0, stddev=0.01)
            # filters3 = tf.random_normal([3, 3, 512, 1024], mean=0.0, stddev=0.01)
            # C2_atrous_conv2d = tf.nn.atrous_conv2d(C2_, filters=filters2, rate=2, padding='SAME')
            # C2_atrous_conv2d = tf.nn.atrous_conv2d(C2_atrous_conv2d, filters=filters3, rate=2, padding='SAME')
            # C2_downsampling = tf.image.resize_bilinear(C2_atrous_conv2d, (C3_shape[1], C3_shape[2]))
            #
            # C4_upsampling = tf.image.resize_bilinear(C4, (C3_shape[1], C3_shape[2]))
            # C4 = C3_atrous_conv2d + C4_upsampling + C2_downsampling

            # C4 = slim.conv2d(C4,
            #                  1024, [5, 5],
            #                  trainable=is_training,
            #                  weights_initializer=cfgs.INITIALIZER,
            #                  activation_fn=None,
            #                  scope='C4_conv5x5')

            C3_shape = tf.shape(end_points_C3['{}/block2/unit_3/bottleneck_v1'.format(scope_name)])
            C4 = tf.image.resize_bilinear(C4, (C3_shape[1], C3_shape[2]))
            _C3 = slim.conv2d(end_points_C3['{}/block2/unit_3/bottleneck_v1'.format(scope_name)],
                              1024, [3, 3],
                              trainable=is_training,
                              weights_initializer=cfgs.INITIALIZER,
                              activation_fn=tf.nn.relu,
                              scope='C3_conv3x3')
            # _C3 = build_inception(end_points_C3['resnet_v1_101/block2/unit_3/bottleneck_v1'], is_training)

            C4 += _C3

        if cfgs.ADD_ATTENTION:
            with tf.variable_scope('build_C4_attention',
                                   regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):
                # tf.summary.image('add_attention_before',
                #                  tf.expand_dims(tf.reduce_mean(C4, axis=-1), axis=-1))

                # SE_C4 = squeeze_excitation_layer(C4, 1024, 16, 'SE_C4', is_training)

                add_heatmap(tf.expand_dims(tf.reduce_mean(C4, axis=-1), axis=-1), 'add_attention_before')
                C4_attention_layer = build_attention(C4, is_training)
                # C4_attention_layer = build_inception_attention(C4, is_training)

                C4_attention = tf.nn.softmax(C4_attention_layer)
                # C4_attention = C4_attention[:, :, :, 1]
                C4_attention = C4_attention[:, :, :, 0]
                C4_attention = tf.expand_dims(C4_attention, axis=-1)
                # tf.summary.image('C3_attention', C4_attention)
                add_heatmap(C4_attention, 'C4_attention')

                C4 = tf.multiply(C4_attention, C4)

                # C4 = SE_C4 * C4
                # tf.summary.image('add_attention_after', tf.expand_dims(tf.reduce_mean(C4, axis=-1), axis=-1))
                add_heatmap(tf.expand_dims(tf.reduce_mean(C4, axis=-1), axis=-1), 'add_attention_after')

    # C4 = tf.Print(C4, [tf.shape(C4)], summarize=10, message='C4_shape')
    if cfgs.ADD_ATTENTION:
        return C4, C4_attention_layer
    else:
        return C4


def restnet_head(input, is_training, scope_name):
    block4 = [resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        C5, _ = resnet_v1.resnet_v1(input,
                                    block4,
                                    global_pool=False,
                                    include_root_block=False,
                                    scope=scope_name)
        # C5 = tf.Print(C5, [tf.shape(C5)], summarize=10, message='C5_shape')
        C5_flatten = tf.reduce_mean(C5, axis=[1, 2], keep_dims=False, name='global_average_pooling')
        # C5_flatten = tf.Print(C5_flatten, [tf.shape(C5_flatten)], summarize=10, message='C5_flatten_shape')

    # global average pooling C5 to obtain fc layers
    return C5_flatten































