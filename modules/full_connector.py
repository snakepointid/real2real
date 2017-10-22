# -*- coding: utf-8 -*-
#/usr/bin/python2
        
import tensorflow as tf
import numpy as np

from real2real.app.params import fullLayerParams

from real2real.layers.common_layers import layer_norm
from pydoc import locate

def multi_layer_perceptron(inputs,output_dim,is_training,is_dropout):
    
        activation_fn = locate(fullLayerParams.activation_fn)

        if fullLayerParams.inputs_reshape:
		
                inputs=tf.reshape(inputs,[-1,int(np.prod(inputs.get_shape()[1:]))])   
                                
        elif len(inputs.get_shape())==3:
                output_dim = 1

        dropout_layer = tf.contrib.layers.dropout(
                                           inputs=inputs,
                                           keep_prob=fullLayerParams.dropout_rate,
                                           is_training=is_dropout)
        
        for layer_idx in range(fullLayerParams.mlp_layers):
                with tf.variable_scope("full_layer{}".format(layer_idx)):
                        hidden_layer    = tf.layers.dense(
                                                        inputs=dropout_layer,
                                                        units=fullLayerParams.hidden_units,
                                                        activation=activation_fn,
                                                        trainable=is_training)

                        dropout_layer   = tf.contrib.layers.dropout(
                                                inputs=hidden_layer,
                                                keep_prob=fullLayerParams.dropout_rate,
                                                is_training=is_dropout)

                        dropout_layer   = layer_norm(dropout_layer)

        logits  = tf.layers.dense(dropout_layer,output_dim, name="output")

        return tf.squeeze(logits)



