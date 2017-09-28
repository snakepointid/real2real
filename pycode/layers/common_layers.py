# -*- coding: utf-8 -*-
#/usr/bin/python2
 

from __future__ import print_function
import tensorflow as tf
from app.params import nlpModelParams,directLayerParams

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
              reuse=None):
   
        with tf.variable_scope(scope, reuse=reuse):
                lookup_table = tf.get_variable('lookup_table',
                                   dtype=tf.float32,
                                   shape=[vocab_size, num_units],
                                   initializer=tf.contrib.layers.xavier_initializer())
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
            scope = "positional_embedding",
            reuse = None):
 

        with tf.variable_scope(scope, reuse = reuse):

                input_one = tf.tile(tf.expand_dims(tf.range(tf.shape(inputs)[1]), 0), [tf.shape(inputs)[0], 1])
                position_block = tf.tile(tf.expand_dims(tf.range(vocab_size), 1), [1, num_units // 2])
                unit_block = tf.tile(tf.expand_dims(tf.range(num_units // 2), 0), [vocab_size, 1])
                rad_block = tf.pow(tf.div(position_block, tf.multiply(10000, 1)), tf.div(unit_block, num_units // 2))
        
                sin_block = tf.sin(tf.cast(rad_block, tf.float32))
                cos_block = tf.cos(tf.cast(rad_block, tf.float32))
                lookup_table = tf.concat([sin_block, cos_block], axis = 1)

                if zero_pad:

                        lookup_table = tf.concat((tf.zeros(shape = [1, num_units]),
                                    lookup_table[1:, :]), 0)
                outputs = tf.nn.embedding_lookup(lookup_table, input_one)
    
                if scale:
                        outputs = outputs * math.sqrt(num_units)

        return outputs

def label_smoothing(inputs, epsilon=0.1):
     
        K = inputs.get_shape().as_list()[-1] # number of channels
        return ((1-epsilon) * inputs) + (epsilon / K)
    
def semantic_position_embedding(inputs,vocab_size,num_units,maxlen,scope):
        with tf.variable_scope(scope):
                encoding = embedding(inputs, 
                                        vocab_size=vocab_size, 
                                        num_units=num_units, 
                                        scale=True,
                                        scope="embedding")
                if not nlpModelParams.flag_position_embed:
                        return encoding
            ## Positional Encoding
                if nlpModelParams.flag_sinusoid:
                        encoding += positional_encoding(inputs,
                                      vocab_size=maxlen, 
                                      num_units=num_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="pe")
                else:
                        encoding += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(inputs)[1]), 0), [tf.shape(inputs)[0], 1]),
                                      vocab_size=maxlen, 
                                      num_units=num_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="pe")
        return encoding

def mlp_layer(inputs,output_dim,mlp_layers,hidden_units,activation_fn,is_training,is_dropout):
	dropout_layer = tf.contrib.layers.dropout(
                                           inputs=inputs,
                                           keep_prob=directLayerParams.dropout_rate,
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
                                                keep_prob=directLayerParams.dropout_rate,
                                                is_training=is_dropout)

                        dropout_layer   = layer_norm(dropout_layer)

        logits  = tf.layers.dense(dropout_layer,output_dim, name="output")
        return tf.squeeze(logits)
