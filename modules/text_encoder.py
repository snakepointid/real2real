# -*- coding: utf-8 -*-
#/usr/bin/python2

import tensorflow as tf

from real2real.layers.conv_layers import strip_conv,max_pool_layer
from real2real.layers.attention_layers import target_attention

from pydoc import locate

def sentence_encoder(inputs,query,layers,multi_cnn_params,scope,is_training,is_dropout,reuse):
        with tf.variable_scope(scope,reuse=reuse):              
                #convolution
                if 'C' in layers:
                        kernel_size,stride_step,conv_layer_num = multi_cnn_params
                        with tf.variable_scope('ml_cov',reuse=reuse):
                                for layer_idx in range(conv_layer_num):
                                        encoding = strip_conv(
                                                       inputs=encoding,
                                                       kernel_size=kernel_size,
                                                       stride_step=stride_step,
                                                       scope_name="conv_layer{}".format(layer_idx),
                                                       is_training = is_training,
                                                       is_dropout=is_dropout)
                #attention                          
                if 'A' in layers:
                        encoding = target_attention(
                                           inputs=encoding,
                                           query=query,
                                           scope_name="target_atten",
                                           is_training=is_training,
                                           is_dropout=is_dropout) #N,m,WD
                        
                if 'P' in layers:
                        encoding = max_pool_layer(encoding)           
        return encoding

 
