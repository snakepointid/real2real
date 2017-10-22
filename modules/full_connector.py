# -*- coding: utf-8 -*-
#/usr/bin/python2
        
import tensorflow as tf
import numpy as np

from real2real.app.params import fullLayerParams

from real2real.layers.common_layers import layer_norm
from pydoc import locate

def multi_layer_perceptron(inputs,output_dim,is_training,is_dropout):
    
        activation_fn = locate(fullLayerParams.activation_fn)

        for layer_idx in range(fullLayerParams.mlp_layers):
                with tf.variable_scope("full_layer{}".format(layer_idx)):                    
                        inputs   = layer_norm(inputs)

                        inputs   = tf.contrib.layers.dropout(
                                                inputs=inputs,
                                                keep_prob=fullLayerParams.dropout_rate,
                                                is_training=is_dropout) 

                        inputs  = tf.layers.dense(
                                                inputs=inputs,
                                                units=fullLayerParams.hidden_units,
                                                activation=activation_fn,
                                                trainable=is_training)
        if fullLayerParams.norm:
                inputs = layer_norm(inputs)

        inputs = tf.contrib.layers.dropout(
                                inputs=inputs,
                                keep_prob=fullLayerParams.dropout_rate,
                                is_training=is_dropout)                

        logits  = tf.layers.dense(inputs,output_dim, name="output")

        return tf.squeeze(logits)



