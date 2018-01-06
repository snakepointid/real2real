# -*- coding: utf-8 -*-
#/usr/bin/python2
 

from __future__ import print_function
import tensorflow as tf
from real2real.app.params import embedLayerParams
import math
from pydoc import locate
def layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
        """Layer normalize the tensor x, averaging over the last dimension."""
        if filters is None:
                filters = x.get_shape()[-1]
        with tf.variable_scope(
                                name, default_name="layer_norm", values=[x], reuse=reuse):
                scale = tf.get_variable( "layer_norm_scale", [filters], initializer=tf.ones_initializer())
                bias  = tf.get_variable( "layer_norm_bias", [filters], initializer=tf.zeros_initializer())
 
                result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result

def layer_norm_compute_python(x, epsilon, scale, bias):
        """Layer norm raw computation."""
        mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * scale + bias



def embedding(inputs, 
              vocab_size,
              embedding_dim=embedLayerParams.embedding_dim,
              zero_pad=True, 
              scope="embedding", 
              is_training=True,
              reuse=None):
   
        with tf.variable_scope(scope, reuse=reuse):
                lookup_table = tf.get_variable('lookup_table',
                                   dtype=tf.float32,
                                   shape=[vocab_size,embedding_dim],
                                   initializer=tf.contrib.layers.xavier_initializer(),
                                   trainable=is_training)
        if zero_pad:
                lookup_table = tf.concat((tf.zeros(shape=[1, embedLayerParams.embedding_dim]), lookup_table[1:, :]), 0)

        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
    
        if embedLayerParams.scale:
                outputs = outputs * (embedLayerParams.embedding_dim ** 0.5) 
        
        return outputs
    

def positional_encoding(inputs,vocab_size,scope = "position_embedding"):

        with tf.name_scope(scope):
                position_block = tf.tile(tf.expand_dims(tf.range(vocab_size), 1), [1, embedLayerParams.embedding_dim // 2])
                unit_block = tf.tile(tf.expand_dims(tf.range(embedLayerParams.embedding_dim // 2), 0), [vocab_size, 1])
                rad_block = tf.pow(tf.div(position_block, tf.multiply(10000, 1)), tf.div(unit_block, embedLayerParams.embedding_dim // 2))
        
                sin_block = tf.sin(tf.cast(rad_block, tf.float32))
                cos_block = tf.cos(tf.cast(rad_block, tf.float32))
                lookup_table = tf.concat([sin_block, cos_block], axis = 1)

                if embedLayerParams.zero_pad:
                        lookup_table = tf.concat((tf.zeros(shape = [1, embedLayerParams.embedding_dim]), lookup_table[1:, :]), 0)
                outputs = tf.nn.embedding_lookup(lookup_table, inputs)
    
                if embedLayerParams.scale:
                        outputs = outputs * math.sqrt(embedLayerParams.embedding_dim)

        return outputs


    



