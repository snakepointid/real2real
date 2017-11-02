# -*- coding: utf-8 -*-
#/usr/bin/python2
import tensorflow as tf
from real2real.layers.common_layers import layer_norm
from real2real.app.params import attentionLayerParams

from pydoc import locate

activation_fn = locate(attentionLayerParams.activation_fn)

def target_attention(inputs,query,scope_name,is_training):
        '''
        inputs N*SL*WD
        query m*QD
        '''
        query_shape = query.get_shape()
        if len(query_shape)==2:
                query = tf.tile(tf.expand_dims(query,0),[tf.shape(inputs)[0],1,1]) # (N, m, QD)
	#apply the layer normalization                
        if attentionLayerParams.norm:
                norm_inputs = layer_norm(inputs)
        else:
                norm_inputs = inputs
	#the keys and values output dim must be the same with the queries dim
        output_dim = int(query.get_shape()[2])
        with tf.variable_scope(scope_name):
                # Linear projections
                K = tf.layers.dense(norm_inputs, output_dim, activation=activation_fn) # (N, SL, QD)
                V = tf.layers.dense(norm_inputs, output_dim, activation=activation_fn) # (N, SL, QD)
                Q = query # (N, m, QD)
                # Multiplication
                weights =  tf.matmul(Q,tf.transpose(K,[0,2,1])) # (N,m,SL)
                #mask
                if attentionLayerParams.zero_pad:
                        mask = tf.sign(tf.reduce_sum(tf.abs(inputs), axis=-1)) #N,SL
                        mask = tf.tile(tf.expand_dims(mask,1),[1,tf.shape(query)[1],1])#N,m,SL
                        paddings = tf.ones_like(weights)*(-2**32+1)
                        weights = tf.where(tf.equal(mask, 0), paddings, weights)
                # Activation
                weights = tf.nn.softmax(weights) # (N,m,SL)
                # Weighted sum
                outputs = tf.matmul(weights, V) # (N,m,QD)

                if attentionLayerParams.direct_cont:
                        # Residual connection
                        outputs += query# (N,m,QD)
                #apply the zero pad mask
                if attentionLayerParams.zero_pad:
                        mask = tf.sign(tf.reduce_sum(tf.abs(inputs), axis=[1,2])) #N 
                        mask = tf.tile(tf.expand_dims(tf.expand_dims(mask,1),1),[1,tf.shape(outputs)[1],tf.shape(outputs)[2]])#(N,m,QD)
                        paddings = tf.zeros_like(outputs)
                        outputs = tf.where(tf.equal(mask, 0), paddings, outputs)

                return outputs

 
 
