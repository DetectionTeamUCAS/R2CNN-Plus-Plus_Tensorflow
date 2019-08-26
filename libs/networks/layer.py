# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs.configs import cfgs
from libs.networks import resnet
from libs.networks import mobilenet_v2


def build_rpn(inputs, num_anchors_per_location, is_training):
    rpn_conv3x3 = slim.conv2d(inputs, 512, [cfgs.KERNEL_SIZE, cfgs.KERNEL_SIZE],
                              trainable=is_training,
                              weights_initializer=cfgs.INITIALIZER,
                              activation_fn=tf.nn.relu,
                              scope='rpn_conv/3x3')
    rpn_cls_score = slim.conv2d(rpn_conv3x3, num_anchors_per_location * 2, [1, 1], stride=1,
                                trainable=is_training, weights_initializer=cfgs.INITIALIZER,
                                activation_fn=None,
                                scope='rpn_cls_score')
    rpn_box_pred = slim.conv2d(rpn_conv3x3, num_anchors_per_location * 4, [1, 1], stride=1,
                               trainable=is_training, weights_initializer=cfgs.BBOX_INITIALIZER,
                               activation_fn=None,
                               scope='rpn_bbox_pred')

    return rpn_cls_score, rpn_box_pred


def roi_pooling(feature_maps, rois, img_shape):
    '''
    Here use roi warping as roi_pooling

    :param featuremaps_dict: feature map to crop
    :param rois: shape is [-1, 4]. [x1, y1, x2, y2]
    :return:
    '''

    with tf.variable_scope('ROI_Warping'):
        img_h, img_w = tf.cast(img_shape[1], tf.float32), tf.cast(img_shape[2], tf.float32)
        N = tf.shape(rois)[0]
        x1, y1, x2, y2 = tf.unstack(rois, axis=1)

        normalized_x1 = x1 / img_w
        normalized_x2 = x2 / img_w
        normalized_y1 = y1 / img_h
        normalized_y2 = y2 / img_h

        normalized_rois = tf.transpose(
            tf.stack([normalized_y1, normalized_x1, normalized_y2, normalized_x2]), name='get_normalized_rois')

        normalized_rois = tf.stop_gradient(normalized_rois)

        cropped_roi_features = tf.image.crop_and_resize(feature_maps, normalized_rois,
                                                        box_ind=tf.zeros(shape=[N, ],
                                                                         dtype=tf.int32),
                                                        crop_size=[cfgs.ROI_SIZE, cfgs.ROI_SIZE],
                                                        name='CROP_AND_RESIZE'
                                                        )
        roi_features = slim.max_pool2d(cropped_roi_features,
                                       [cfgs.ROI_POOL_KERNEL_SIZE, cfgs.ROI_POOL_KERNEL_SIZE],
                                       stride=cfgs.ROI_POOL_KERNEL_SIZE)

    return roi_features


def build_fastrcnn(feature_to_cropped, rois, img_shape, base_network_name, is_training):
    with tf.variable_scope('Fast-RCNN'):
        # 1. ROI Pooling
        with tf.variable_scope('rois_pooling'):
            pooled_features = roi_pooling(feature_maps=feature_to_cropped, rois=rois, img_shape=img_shape)

        # 2. inferecne rois in Fast-RCNN to obtain fc_flatten features
        if base_network_name.startswith('resnet'):
            fc_flatten = resnet.restnet_head(input=pooled_features,
                                             is_training=is_training,
                                             scope_name=base_network_name)
        elif base_network_name.startswith('MobilenetV2'):
            fc_flatten = mobilenet_v2.mobilenetv2_head(inputs=pooled_features,
                                                       is_training=is_training)
        else:
            raise NotImplementedError('only support resnet and mobilenet')

        # 3. cls and reg in Fast-RCNN
        with tf.variable_scope('horizen_branch'):
            with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):

                cls_score_h = slim.fully_connected(fc_flatten,
                                                   num_outputs=cfgs.CLASS_NUM + 1,
                                                   weights_initializer=cfgs.INITIALIZER,
                                                   activation_fn=None, trainable=is_training,
                                                   scope='cls_fc_h')

                bbox_pred_h = slim.fully_connected(fc_flatten,
                                                   num_outputs=(cfgs.CLASS_NUM + 1) * 4,
                                                   weights_initializer=cfgs.BBOX_INITIALIZER,
                                                   activation_fn=None, trainable=is_training,
                                                   scope='reg_fc_h')

                # for convient. It also produce (cls_num +1) bboxes

                cls_score_h = tf.reshape(cls_score_h, [-1, cfgs.CLASS_NUM + 1])
                bbox_pred_h = tf.reshape(bbox_pred_h, [-1, 4 * (cfgs.CLASS_NUM + 1)])

        with tf.variable_scope('rotation_branch'):
            with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):

                cls_score_r = slim.fully_connected(fc_flatten,
                                                   num_outputs=cfgs.CLASS_NUM + 1,
                                                   weights_initializer=cfgs.INITIALIZER,
                                                   activation_fn=None, trainable=is_training,
                                                   scope='cls_fc_r')

                bbox_pred_r = slim.fully_connected(fc_flatten,
                                                   num_outputs=(cfgs.CLASS_NUM + 1) * 5,
                                                   weights_initializer=cfgs.BBOX_INITIALIZER,
                                                   activation_fn=None, trainable=is_training,
                                                   scope='reg_fc_r')
                # for convient. It also produce (cls_num +1) bboxes
                cls_score_r = tf.reshape(cls_score_r, [-1, cfgs.CLASS_NUM + 1])
                bbox_pred_r = tf.reshape(bbox_pred_r, [-1, 5 * (cfgs.CLASS_NUM + 1)])

        return bbox_pred_h, cls_score_h, bbox_pred_r, cls_score_r


