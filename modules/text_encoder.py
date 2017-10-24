# -*- coding: utf-8 -*-
#/usr/bin/python2

import tensorflow as tf

from real2real.layers.conv_layers import multiLayer_conv_strip,conv_to_full_layer
from real2real.layers.attention_layers import target_attention

from real2real.app.params import textModuleParams
from pydoc import locate

def sentence_encoder(inputs,query,multi_cnn_params,scope,is_training,is_dropout,reuse):
        encoding = inputs
        with tf.variable_scope(scope,reuse=reuse):              
                #convolution
                if textModuleParams.stride_cnn:
                        encoding = multiLayer_conv_strip(
                                           inputs=encoding,
                                           multi_cnn_params=multi_cnn_params,
                                           scope_name='ml_cnn',
                                           is_training=is_training,
                                           is_dropout=is_dropout)

                if textModuleParams.target_atten:
                        encoding = target_attention(
                                           inputs=encoding,
                                           query=query,
                                           scope_name="target_atten",
                                           is_training=is_training,
                                           is_dropout=is_dropout) #N,m,WD                 
        return encoding

 
