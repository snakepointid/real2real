# -*- coding: utf-8 -*-
#/usr/bin/python2

import tensorflow as tf
from real2real.layers.conv_layers import multiLayer_conv_strip,conv1d_to_full_layer
from real2real.layers.common_layers import semantic_position_embedding,embedding
from real2real.layers.attention_layers import multi_hot_attention
from pydoc import locate

def short_text_conv_encoder(inputs,vocab_size,num_units,zero_pad,scale,maxlen,scope,is_training,is_dropout,reuse):
        with tf.variable_scope(scope,reuse=reuse):
                embed = semantic_position_embedding(
                                           inputs=inputs,
                                           vocab_size=vocab_size,
                                           num_units=num_units,
                                           is_training=is_training,
                                           zero_pad=zero_pad,
                                           scale=scale,
                                           maxlen=maxlen,
                                           scope='embedding')
                #convolution
                conv_out = multiLayer_conv_strip(
                                    inputs=embed,
                                    kernel_size=5,
                                    conv_layer_num=3,
                                    stride_step=2,
                                    scope_name='ml_cnn',
                                    zero_pad=zero_pad,
                                    is_training=is_training,
                                    is_dropout=is_dropout)
                         
                full_layer = conv1d_to_full_layer(
                                    inputs=conv_out,
                                    scope_name="conv2full",
                                    is_training=is_training)
        return full_layer