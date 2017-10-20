# -*- coding: utf-8 -*-
#/usr/bin/python2
 

from __future__ import print_function
import tensorflow as tf
from real2real.app.params import embedLayerParams,baseLayerParams
import math
from pydoc import locate
def layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
        """Layer normalize the tensor x, averaging over the last dimension."""
        if filters is None:
                filters = x.get_shape()[-1]
        with tf.variable_scope(
                                name, default_name="layer_norm", values=[x], reuse=reuse):
                scale = tf.get_variable(
                                        "layer_norm_scale", [filters], initializer=tf.ones_initializer())
                bias  = tf.get_variable(
                                        "layer_norm_bias", [filters], initializer=tf.zeros_initializer())
 
                result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result

def layer_norm_compute_python(x, epsilon, scale, bias):
        """Layer norm raw computation."""
        mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * scale + bias

def noam_norm(x, epsilon=1.0, name=None):
        """One version of layer normalization."""
        with tf.name_scope(name, default_name="noam_norm", values=[x]):
                shape = x.get_shape()
                ndims = len(shape)
                result = tf.nn.l2_normalize(x, ndims - 1, epsilon=epsilon)*tf.sqrt(tf.to_float(shape[-1]))
                return result

def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True, 
              scale=True,
              scope="embedding", 
              is_training=True,
              reuse=None):
   
        with tf.variable_scope(scope, reuse=reuse):
                lookup_table = tf.get_variable('lookup_table',
                                   dtype=tf.float32,
                                   shape=[vocab_size, num_units],
                                   initializer=tf.contrib.layers.xavier_initializer(),
                                   trainable=is_training)
        if zero_pad:
                lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
    
        if scale:
                outputs = outputs * (num_units ** 0.5) 
        
        return outputs
    

def positional_encoding(inputs,
            vocab_size,
            num_units,
            zero_pad = True,
            scale = True,
            scope = "position_embedding",
            reuse = None):
 
        with tf.variable_scope(scope, reuse = reuse):
                position_block = tf.tile(tf.expand_dims(tf.range(vocab_size), 1), [1, num_units // 2])
                unit_block = tf.tile(tf.expand_dims(tf.range(num_units // 2), 0), [vocab_size, 1])
                rad_block = tf.pow(tf.div(position_block, tf.multiply(10000, 1)), tf.div(unit_block, num_units // 2))
        
                sin_block = tf.sin(tf.cast(rad_block, tf.float32))
                cos_block = tf.cos(tf.cast(rad_block, tf.float32))
                lookup_table = tf.concat([sin_block, cos_block], axis = 1)

                if zero_pad:
                        lookup_table = tf.concat((tf.zeros(shape = [1, num_units]), lookup_table[1:, :]), 0)
                outputs = tf.nn.embedding_lookup(lookup_table, inputs)
    
                if scale:
                        outputs = outputs * math.sqrt(num_units)

        return outputs

def label_smoothing(inputs, epsilon=0.1):
        K = inputs.get_shape().as_list()[-1] # number of channels
        return ((1-epsilon) * inputs) + (epsilon / K)
    
def semantic_position_embedding(inputs,vocab_size,num_units,maxlen,zero_pad,scale,is_training,scope,reuse=None):
        with tf.variable_scope(scope,reuse=reuse):
                encoding = embedding(inputs=inputs, 
                                     vocab_size=vocab_size, 
                                     num_units=num_units, 
                                     zero_pad=zero_pad,
                                     scale=scale,
                                     reuse=reuse,
                                     is_training=is_training,
                                     scope="token")

                if not embedLayerParams.flag_position_embed:
                        return encoding
                ## Positional Encoding
                position_code = tf.tile(tf.expand_dims(tf.to_int64(tf.range(tf.shape(inputs)[1])+1), 0), [tf.shape(inputs)[0], 1])
                paddings = tf.zeros_like(inputs)
                position_code = tf.where(tf.equal(inputs, 0), paddings, position_code)

                if embedLayerParams.flag_sinusoid:
                        encoding += positional_encoding(
                                        inputs=position_code,
                                        vocab_size=maxlen+1, 
                                        num_units=num_units, 
                                        zero_pad=zero_pad, 
                                        scale=scale,
                                        scope="position")
                else:
                        encoding += embedding(
                                        inputs=position_code,
                                        vocab_size=maxlen+1, 
                                        num_units=num_units, 
                                        zero_pad=zero_pad, 
                                        scale=scale,
                                        reuse=reuse,
                                        is_training=is_training,
                                        scope="position")
        return encoding


