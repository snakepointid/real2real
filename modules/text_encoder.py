# -*- coding: utf-8 -*-
#/usr/bin/python2

import tensorflow as tf
from real2real.layers.conv_layers import multiLayer_conv_strip,conv1d_to_full_layer
from real2real.layers.common_layers import semantic_position_embedding,embedding
from real2real.layers.attention_layers import target_attention
from pydoc import locate

def text_conv_encoder(inputs,vocab_size,multi_cnn_params,scope,is_training,is_dropout,reuse):
        with tf.variable_scope(scope,reuse=reuse):
                if len(inputs.get_shape())==2:
                        embed = semantic_position_embedding(
                                           inputs=inputs,
                                           vocab_size=vocab_size,
                                           is_training=is_training,
                                           scope='embedding')
                else:
                        embed = inputs                 
                #convolution
                conv_out = multiLayer_conv_strip(
                                           inputs=embed,
                                           multi_cnn_params=multi_cnn_params,
                                           scope_name='ml_cnn',
                                           is_training=is_training,
                                           is_dropout=is_dropout)
                         
                full_layer = conv1d_to_full_layer(
                                           inputs=conv_out,
                                           scope_name="conv2full",
                                           is_training=is_training)
        return full_layer

def text_atten_encoder(inputs,query,vocab_size,multi_cnn_params,maxlen,scope,is_training,is_dropout,reuse):
        with tf.variable_scope(scope,reuse=reuse):
                if len(inputs.get_shape())==2:
                        embed = semantic_position_embedding(
                                           inputs=inputs,
                                           vocab_size=vocab_size,
                                           is_training=is_training,
                                           scope='embedding')
                else:
                        embed = inputs
                #convolution
                
                conv_out = multiLayer_conv_strip(
                                           inputs=embed,
                                           multi_cnn_params=multi_cnn_params,
                                           scope_name='ml_cnn',
                                           is_training=is_training,
                                           is_dropout=is_dropout)

                atten_layer = target_attention(
                                           inputs=conv_out,
                                           query=query,
                                           scope_name="target_atten",
                                           is_training=is_training) #N,m,WD
                 
        return atten_layer
