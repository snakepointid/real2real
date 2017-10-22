# -*- coding: utf-8 -*-
#/usr/bin/python2
        
import tensorflow as tf
import numpy as np

from real2real.app.params import fullLayerParams

from real2real.layers.common_layers import layer_norm
from pydoc import locate

activation_fn = locate(fullLayerParams.activation_fn)

def final_mlp_encoder(inputs,output_dim,is_training,is_dropout):
        static_shape = inputs.get_shape()
        if len(static_shape)==3:
                inputs = tf.reshape(inputs,[-1,int(static_shape[1])*int(static_shape[2])])
                
        for layer_idx in range(fullLayerParams.mlp_layers):
                with tf.variable_scope("full_layer{}".format(layer_idx)):   
                        if fullLayerParams.norm:                 
                                inputs  = layer_norm(inputs)                 
                        inputs  = tf.layers.dense(
                                                inputs=inputs,
                                                units=fullLayerParams.hidden_units,
                                                activation=activation_fn,
                                                trainable=is_training)
                        inputs  = tf.contrib.layers.dropout(
                                                inputs=inputs,
                                                keep_prob=fullLayerParams.dropout_rate,
                                                is_training=is_dropout)

        logits = tf.layers.dense(inputs,output_dim, name="output")

        return tf.squeeze(logits)



