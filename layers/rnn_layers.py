# -*- coding: utf-8 -*-
#/usr/bin/python2
import tensorflow as tf
from real2real.layers.common_layers import layer_norm
from real2real.app.params import rnnLayerParams

from pydoc import locate

activation_fn = locate(rnnLayerParams.activation_fn)

def uni_lstm(inputs,reuse,is_training):

        seq_length = tf.to_int32(tf.reduce_sum(tf.sign(tf.reduce_sum(tf.abs(inputs), axis=-1)),1))#N,SL
 
        basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnnLayerParams.hidden_units,activation=activation_fn,reuse=reuse)
        outputs, states = tf.nn.dynamic_rnn(basic_cell, inputs, dtype=tf.float32,sequence_length=seq_length)

        return outputs#N,SL,D

def bi_lstm(inputs,reuse,is_training):

        seq_length = tf.to_int32(tf.reduce_sum(tf.sign(tf.reduce_sum(tf.abs(inputs), axis=-1)),1))#N,SL
 
        cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=rnnLayerParams.hidden_units,activation=activation_fn,reuse=reuse)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=rnnLayerParams.hidden_units,activation=activation_fn,reuse=reuse)

        outputs, states =  tf.nn.bidirectional_dynamic_rnn(
                                                        cell_fw=cell_fw,
                                                        cell_bw=cell_bw,
                                                        inputs=inputs,
                                                        sequence_length=seq_length,
                                                        dtype=tf.float32)
        return tf.concat(outputs, 2)#N,SL,2*D
 
 
