# -*- coding: utf-8 -*-
#/usr/bin/python2
import tensorflow as tf
from real2real.layers.common_layers import layer_norm
from real2real.app.params import attentionLayerParams

from pydoc import locate

def target_attention_3d(inputs,query,scope_name,is_training):
        '''
        inputs N*SL*WD
        query N*m*QD
        '''
        query_shape = query.get_shape()

        if len(query_shape)==2:
                query = tf.expand_dims(query,1) # (N, 1, QD)
        elif len(query_shape)==3:
                pass
        else:
                raise ValueError("the rank of query must be 2 or 3")

        activation_fn = locate(attentionLayerParams.activation_fn)
        out_dim = int(query.get_shape()[2])
        with tf.variable_scope(scope_name):
                # Linear projections
                K = tf.layers.dense(inputs, out_dim, activation=activation_fn) # (N, SL, QD)
                V = tf.layers.dense(inputs, out_dim, activation=activation_fn) # (N, SL, QD)
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
                if attentionLayerParams.norm:
                        # Normalize
                        outputs = layer_norm(outputs) # (N,m,QD)
                #sentence mask
                if attentionLayerParams.zero_pad:
                        mask = tf.sign(tf.reduce_sum(tf.abs(inputs), axis=[1,2])) #N 
                        mask = tf.tile(tf.expand_dims(tf.expand_dims(mask,1),1),[1,tf.shape(outputs)[1],tf.shape(outputs)[2]])#(N,m,QD)
                        paddings = tf.zeros_like(outputs)
                        outputs = tf.where(tf.equal(mask, 0), paddings, outputs)
                return outputs

def target_attention_4d(inputs,query,scope_name,is_training):
        '''
        inputs N*SL*m*WD
        query N*m*QD
        '''
        query_shape = query.get_shape()

        if len(query_shape)==3:
                reshaped_query = tf.tile(tf.expand_dims(query,1),[1,tf.shape(inputs)[1],1,1])# (N, SL,m, QD)
        elif len(query_shape)==4:
                pass
        else:
                raise ValueError("the rank of query must be 3 or 4")

        activation_fn = locate(attentionLayerParams.activation_fn)

        out_dim = int(reshaped_query.get_shape()[3])

        with tf.variable_scope(scope_name):
                # Linear projections
                K = tf.layers.dense(inputs, out_dim, activation=activation_fn) # (N, SL, m, QD)
                V = tf.layers.dense(inputs, out_dim, activation=activation_fn) # (N, SL, m, QD)
                Q = reshaped_query # (N,SL, m, QD)

                # Multiplication
                weights =  tf.reduce_sum(tf.multiply(Q,K),3) # (N, SL, m)
                #mask
                if attentionLayerParams.zero_pad:
                        mask = tf.sign(tf.reduce_sum(tf.abs(inputs), axis=[2,3])) #N,SL
                        mask = tf.tile(tf.expand_dims(mask,2),[1,1,tf.shape(reshaped_query)[2]])#N,SL,m
                        paddings = tf.ones_like(weights)*(-2**32+1)
                        weights = tf.where(tf.equal(mask, 0), paddings, weights)
                # Activation
                weights = tf.nn.softmax(weights,1) # (N, SL, m)
                weights = tf.tile(tf.expand_dims(weights,3),tf.shape(reshaped_query)[-1])# (N, SL, m,QD)
                # Weighted sum
                outputs = tf.reduce_sum(tf.multiply(weights,V),1) # (N,m,QD)

                if attentionLayerParams.direct_cont:
                        # Residual connection
                        outputs += query# (N,m,QD)
                if attentionLayerParams.norm:
                        # Normalize
                        outputs = layer_norm(outputs) # (N,m,QD)

                #sentence mask
                if attentionLayerParams.zero_pad:
                        mask = tf.sign(tf.reduce_sum(tf.abs(inputs), axis=[1,2,3])) #N 
                        mask = tf.tile(tf.expand_dims(tf.expand_dims(mask,1),1),[1,tf.shape(outputs)[1],tf.shape(outputs)[2]])#(N,m,QD)
                        paddings = tf.zeros_like(outputs)
                        outputs = tf.where(tf.equal(mask, 0), paddings, outputs)
                        
                return outputs

 
