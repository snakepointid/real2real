# -*- coding: utf-8 -*-
#/usr/bin/python2
 

from __future__ import print_function
import tensorflow as tf
from real2real.layers.common_layers import *
from real2real.layers.conv_layers import strip_conv,direct_conv
from real2real.app.params import attentionLayerParams

def self_attention(encoding,is_training,is_dropout):
        with tf.variable_scope("encoder"):  
                ## Dropout
                encoding = tf.contrib.layers.dropout(encoding, 
                                            keep_prob=attentionLayerParams.dropout_rate, 
                                            is_training=is_dropout)
                
                ## Blocks
                for i in range(attentionLayerParams.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        encoding = multihead_attention(queries=encoding, 
                                                        keys=encoding, 
                                                        num_heads=attentionLayerParams.num_heads, 
                                                        dropout_rate=attentionLayerParams.dropout_rate,
                                                        is_training=is_training,
                                                        causality=False,
                                                        is_dropout=is_dropout)
                        
                        ### Feed Forward
                        encoding = feedforward(encoding, num_units=[4*attentionLayerParams.hidden_units, attentionLayerParams.hidden_units])
        return encoding

def multi_hot_attention(inputs,query,scope_name,is_training):
        '''
        inputs N*SL*WD
        query N*QD
        '''
        query_shape = query.get_shape()

        if len(query_shape)==2:
                query = tf.expand_dims(query,1) # (N, 1, QD)
        elif len(query_shape)==3:
                pass
        else:
                raise ValueError("the rank of query must be 2 or 3")

        activation_fn = locate(attentionLayerParams.activation_fn)
        with tf.variable_scope(scope_name):
                # Linear projections
                K = tf.layers.dense(inputs, query.get_shape()[1], activation=activation_fn) # (N, SL, QD)
                V = tf.layers.dense(inputs, query.get_shape()[1], activation=activation_fn) # (N, SL, QD)
                Q = query # (N, m, QD)
                # Multiplication
                weights =  tf.matmul(Q,tf.transpose(K,[0,2,1])) # (N,m,SL)
                # Activation
                weights = tf.nn.softmax(weights) # (N,m,SL)
                # Weighted sum
                outputs = tf.matmul(weights, V) # (N,m,QD)
                # Residual connection
                outputs += queries# (N,m,QD)
                # Normalize
                outputs = layer_norm(outputs) # (N,m,QD)
                return outputs

def enc_dec_attention(decoding,encoding,is_training,is_dropout):
        # Decoder
        with tf.variable_scope("decoder"):
                ## Dropout
                decoding = tf.contrib.layers.dropout(decoding, 
                                            keep_prob=attentionLayerParams.dropout_rate, 
                                            is_training=is_dropout)
                
                ## Blocks
                for i in range(attentionLayerParams.num_blocks):
                        with tf.variable_scope("num_blocks_{}".format(i)):
                                ## Multihead Attention ( self-attention)
                                decoding = multihead_attention(queries=decoding, 
                                                        keys=decoding, 
                                                        num_heads=attentionLayerParams.num_heads, 
                                                        dropout_rate=attentionLayerParams.dropout_rate,
                                                        is_training=is_training,
                                                        causality=True, 
                                                        scope="self_attention",
                                                        is_dropout=is_dropout)
                        
                                ## Multihead Attention ( vanilla attention)
                                decoding = multihead_attention(queries=decoding, 
                                                        keys=encoding,  
                                                        num_heads=attentionLayerParams.num_heads,
                                                        dropout_rate=attentionLayerParams.dropout_rate,
                                                        is_training=is_training, 
                                                        causality=False,
                                                        scope="vanilla_attention",
                                                        is_dropout=is_dropout)
                        
                                ## Feed Forward
                                decoding = feedforward(decoding, num_units=[4*attentionLayerParams.hidden_units, attentionLayerParams.hidden_units])
        return decoding

        
def multihead_attention(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        is_dropout=None,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None):
 
        with tf.variable_scope(scope, reuse=reuse):
                # Set the fall back option for num_units
                if num_units is None:
                        num_units = queries.get_shape()[-1]
        
                # Linear projections
                Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
                K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
                V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        
                # Split and concat
                Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
                K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
                V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

                # Multiplication
                outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
                # Scale
                outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
                # Key Masking
                key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
                key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
                key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
                paddings = tf.ones_like(outputs)*(-2**32+1)
                outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
                # Causality = Future blinding
                if causality:
                        diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
                        tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
                        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
                        paddings = tf.ones_like(masks)*(-2**32+1)
                        outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
                # Activation
                outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
                # Query Masking
                query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
                query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
                query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
                outputs *= query_masks # broadcasting. (N, T_q, C)
          
                # Dropouts
                outputs = tf.contrib.layers.dropout(
                                            inputs=outputs,
                                            keep_prob=dropout_rate,
                                            is_training=is_dropout)
               
                # Weighted sum
                outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
                # Restore shape
                outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
                # Residual connection
                outputs += queries
              
                # Normalize
                outputs = layer_norm(outputs) # (N, T_q, C)
 
        return outputs

def feedforward(encoder_inputs, 
                num_units=[2048, 512],
                scope="forward", 
                reuse=None):
     
        with tf.variable_scope(scope, reuse=reuse):
                # Inner layer
                params = {"inputs": encoder_inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
                outputs = tf.layers.conv1d(**params)
        
                # Readout layer
                params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
                outputs = tf.layers.conv1d(**params)
        
                # Residual connection
                outputs += encoder_inputs
        
                # Normalize
                outputs = layer_norm(outputs)
    
        return outputs
