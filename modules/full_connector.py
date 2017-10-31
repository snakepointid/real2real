# -*- coding: utf-8 -*-
#/usr/bin/python2
        
import tensorflow as tf
import numpy as np

from real2real.app.params import fullConnectModuleParams

from real2real.layers.common_layers import layer_norm

from pydoc import locate

#get the activate function
activation_fn = locate(fullConnectModuleParams.activation_fn)
'''
	this is a base full connect layers
'''
def final_mlp_encoder(inputs,output_dim,is_training,is_dropout):
	#get the static shape of the input tensor
        static_shape = inputs.get_shape()
	#if the rank of the inputs is equal to 3,then reshape the inputs
        if fullConnectModuleParams.input_reshape and len(static_shape)==3:
                inputs = tf.reshape(inputs,[-1,int(static_shape[1])*int(static_shape[2])])
	#or,set the output dim to be 1 and squeeze the outputs
        if not fullConnectModuleParams.input_reshape and len(static_shape)==3:
                output_dim = 1
	#apply the dropout to inputs
        inputs  = tf.contrib.layers.dropout(
                                           inputs=inputs,
                                           keep_prob=fullConnectModuleParams.dropout_rate,
                                           is_training=is_dropout) 
	#a normal multi-layer percentron
        for layer_idx in range(fullConnectModuleParams.mlp_layers):
                with tf.variable_scope("full_layer{}".format(layer_idx)):   
                        if fullConnectModuleParams.norm:                 
                                inputs  = layer_norm(inputs)                 
                        inputs  = tf.layers.dense(
                                                inputs=inputs,
                                                units=fullConnectModuleParams.hidden_units,
                                                activation=activation_fn,
                                                trainable=is_training)
                        inputs  = tf.contrib.layers.dropout(
                                                inputs=inputs,
                                                keep_prob=fullConnectModuleParams.dropout_rate,
                                                is_training=is_dropout)
	#final projection
        logits = tf.layers.dense(inputs,output_dim, name="output")
        return tf.squeeze(logits)



