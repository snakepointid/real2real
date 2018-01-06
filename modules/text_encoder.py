# -*- coding: utf-8 -*-
#/usr/bin/python2

import tensorflow as tf

from real2real.layers.pool_layers import *
from real2real.layers.conv_layers import strip_conv
from real2real.layers.attention_layers import simple_attention,semantic_attention
from real2real.layers.rnn_layers import bi_lstm,uni_lstm

from real2real.app.params import *
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
                if 'C' in textModuleParams.layers:
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
                if 'R' in textModuleParams.layers:
                        encoding = bi_lstm(
                                           inputs=encoding, 
                                           reuse=reuse,
                                           is_training=is_training) 
                #target attention                          
                if 'A' in textModuleParams.layers:
                        encoding,_,_ = simple_attention(
                                           inputs=encoding,
                                           query=query,
                                           scope_name="label_attention",
                                           is_training=is_training) #N,m,WD
    #simple pool layer        
                if 'P' in textModuleParams.layers:
                        encoding = max_pool_layer(encoding)           
        return encoding


def keyword_encoder(inputs,query,scope,is_training,is_dropout,reuse):
        #apply the dropout to the inputs
        encoding = tf.contrib.layers.dropout(
                                           inputs=inputs,
                                           keep_prob=0.5,
                                           is_training=is_dropout) 
        with tf.variable_scope(scope,reuse=reuse):
                if keywordModuleParams.types==1:
                        encoding,weights,weights_logits = simple_attention(
                                           inputs=encoding,
                                           query=query,
                                           scope_name="global_attention",
                                           is_training=is_training) #N,m,WD
                
                if keywordModuleParams.types==2:
                        query = avg_pool_layer(
                                           inputs=encoding)

                        encoding,weights,weights_logits = semantic_attention(
                                           inputs=encoding,
                                           query=query,
                                           scope_name="semantic_attention",
                                           is_training=is_training) #N,m,WD

                if keywordModuleParams.types==3:
                        query,_,_ = simple_attention(
                                           inputs=encoding,
                                           query=query,
                                           scope_name="global_attention",
                                           is_training=is_training) #N,1,WD

                        encoding,weights,weights_logits = semantic_attention(
                                           inputs=encoding,
                                           query=query,
                                           scope_name="semantic_attention",
                                           is_training=is_training) #N,m,WD

        return tf.squeeze(encoding,[1]),tf.squeeze(weights,[1]),tf.squeeze(weights_logits,[1])

 
