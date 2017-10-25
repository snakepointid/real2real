# -*- coding: utf-8 -*-
#/usr/bin/python2

import tensorflow as tf

from real2real.layers.conv_layers import strip_conv,max_pool_layer
from real2real.layers.attention_layers import target_attention
 
from real2real.app.params import textModuleParams
from pydoc import locate

def sentence_encoder(inputs,query,multi_cnn_params,scope,is_training,is_dropout,reuse):
        encoding = tf.contrib.layers.dropout(
                                           inputs=inputs,
                                           keep_prob=textModuleParams.dropout_rate,
                                           is_training=is_dropout)
        with tf.variable_scope(scope,reuse=reuse):              
                #convolution
                if textModuleParams.stride_cnn:
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
                if textModuleParams.target_atten:
                        encoding = target_attention(
                                           inputs=encoding,
                                           query=query,
                                           scope_name="target_atten",
                                           is_training=is_training,
                                           is_dropout=is_dropout) #N,m,WD
                        
                if textModuleParams.max_pool:
                        encoding = max_pool_layer(encoding)           
        return encoding

 
