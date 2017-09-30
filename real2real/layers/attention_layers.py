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

def conv_attention_conv(inputs,query_length,scope_name,is_training,is_dropout):
        with tf.variable_scope("encoder"):
                ## Dropout
                inputs = tf.contrib.layers.dropout(inputs,
                                            keep_prob=attentionLayerParams.dropout_rate,
                                            is_training=is_dropout)


                with tf.variable_scope("num_blocks_{}".format(i)):
                        conv_out=direct_conv(
                                            inputs=inputs,
                                            scope_name='before_atten',
                                            is_training=is_training)  
                        atten_out=position_attention_1d(
                                                        inputs=conv_out,
                                                        query_length=query_length,
                                                        scope_name='posi_atten',
                                                        is_training=is_training,
                                                        is_dropout=is_dropout)
                        conv_out=direct_conv(
                                            inputs=atten_out,
                                            scope_name='after_atten',
                                            is_training=is_training)
        return conv_out
def position_attention_1d(inputs,query_length,scope_name,is_training,is_dropout):
        '''
            inputs N * L * E

        '''
        static_shape  = inputs.get_shape()
        with tf.variable_scope(scope_name):
                inputs = tf.contrib.layers.dropout(inputs,
                                            keep_prob=attentionLayerParams.dropout_rate,
                                            is_training=is_dropout)
                add_position_embed = tf.get_variable(
                                                    'add_position', 
                                                    shape=[1,static_shape[1],static_shape[2]],
                                                    trainable=is_training)
                query_position_embed = tf.get_variable(
                                                    'query_position', 
                                                    shape=[1,query_length,2*static_shape[2]],
                                                    trainable=is_training)

                #ops
                keys = tf.layers.dense(inputs, static_shape[2], activation=tf.nn.relu) #N,L,E
                values = tf.layers.dense(inputs, static_shape[2], activation=tf.nn.relu) #N,L,E
                #augment
                add_position_embed = tf.tile(add_position_embed,[tf.shape(inputs)[0],1,1])
                query_position_embed = tf.tile(query_position_embed,[tf.shape(inputs)[0],1,1])#N,QL,2*E
                #combine
                keys = tf.concat([keys,add_position_embed],2) # N,L,2*E
                outputs = tf.matmul(query_position_embed, tf.transpose(keys, [0, 2, 1])) #N,QL,L

                key_masks = tf.sign(tf.abs(tf.reduce_sum(inputs, axis=-1))) # (N, L)
                key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1,query_length,1]) # N,QL,L

                paddings = tf.ones_like(outputs)*(-2**32+1)
                outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # N,QL,L
                outputs = tf.nn.softmax(outputs) # N,QL,L

                outputs = tf.matmul(outputs, values) #N,QL,E
                # Normalize
                outputs = layer_norm(outputs) # N,QL,E

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
