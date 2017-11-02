# -*- coding: utf-8 -*-
#/usr/bin/python2

import tensorflow as tf

from real2real.layers.conv_layers import strip_conv,max_pool_layer
from real2real.layers.attention_layers import target_attention
from real2real.layers.rnn_layers import bi_lstm
'''
	this is a general text encoding module
	it can be consisted by three parts: conv,attention and pool layers
'''
def sentence_encoder(inputs,query,layers,multi_cnn_params,scope,is_training,is_dropout,reuse):
        #apply the dropout to the inputs
        encoding = tf.contrib.layers.dropout(
                                           inputs=inputs,
                                           keep_prob=0.5,
                                           is_training=is_dropout) 
        with tf.variable_scope(scope,reuse=reuse):              
                #multi layers stride convolution without pool
                if 'C' in layers:
                        kernel_size,stride_step,conv_layer_num = multi_cnn_params
                        with tf.variable_scope('ml_cov',reuse=reuse):
                                for layer_idx in range(conv_layer_num):
                                        encoding = strip_conv(
                                                       inputs=encoding,
                                                       kernel_size=kernel_size,
                                                       stride_step=stride_step,
                                                       scope_name="conv_layer{}".format(layer_idx),
                                                       is_training = is_training)
                #rnn 
                if 'R' in layers:
                        encoding = bi_lstm(
                                           inputs=encoding, 
                                           scope_name="bidirect_lstm",
                                           is_training=is_training) 
                #target attention                          
                if 'A' in layers:
                        encoding = target_attention(
                                           inputs=encoding,
                                           query=query,
                                           scope_name="target_atten",
                                           is_training=is_training) #N,m,WD
		#simple pool layer        
                if 'P' in layers:
                        encoding = max_pool_layer(encoding)           
        return encoding

 
