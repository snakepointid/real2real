# -*- coding: utf-8 -*-
#/usr/bin/python2
        
import tensorflow as tf
from real2real.app.params import baseLayerParams
from real2real.layers.common_layers import layer_norm
from pydoc import locate

def multi_layer_perceptron(inputs,output_dim,mlp_layers,hidden_units,activation_fn,is_training,is_dropout):
        activation_fn = locate(activation_fn)
        dropout_layer = tf.contrib.layers.dropout(
                                           inputs=inputs,
                                           keep_prob=baseLayerParams.dropout_rate,
                                           is_training=is_dropout)
        for layer_idx in range(mlp_layers):
                with tf.variable_scope("full_layer{}".format(layer_idx)):
                        hidden_layer    = tf.layers.dense(
                                                        inputs=dropout_layer,
                                                        units=hidden_units,
                                                        activation=activation_fn,
                                                        trainable=is_training)

                        dropout_layer   = tf.contrib.layers.dropout(
                                                inputs=hidden_layer,
                                                keep_prob=baseLayerParams.dropout_rate,
                                                is_training=is_dropout)

                        dropout_layer   = layer_norm(dropout_layer)

        logits  = tf.layers.dense(dropout_layer,output_dim, name="output")

        return tf.squeeze(logits)
