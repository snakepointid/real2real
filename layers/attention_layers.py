# -*- coding: utf-8 -*-
#/usr/bin/python2
import tensorflow as tf
from real2real.layers.common_layers import layer_norm
from real2real.app.params import attentionLayerParams

from pydoc import locate

activation_fn = locate(attentionLayerParams.activation_fn)

def simple_attention(inputs,query,scope_name,is_training):
        with tf.variable_scope(scope_name):
                '''
                inputs N*SL*WD
                query m*QD
                '''
                query_shape = query.get_shape()
                if len(query_shape)==2:
                        query = tf.tile(tf.expand_dims(query,0),[tf.shape(inputs)[0],1,1]) # (N, m, QD)
                #apply the layer normalization                
                if attentionLayerParams.norm:
                        norm_inputs = layer_norm(inputs,name="inputs")
                else:
                        norm_inputs = inputs
                #the keys and values output dim must be the same with the queries dim
                output_dim = int(query.get_shape()[2])
                # Linear projections
                K = tf.layers.dense(norm_inputs, output_dim, activation=activation_fn,name="key") # (N, SL, QD)
                V = tf.layers.dense(norm_inputs, output_dim, activation=activation_fn,name="value") # (N, SL, QD)
                Q = query # (N, m, QD)
                # Multiplication
                weights_logits = tf.matmul(Q,tf.transpose(K,[0,2,1])) # (N,m,SL)
                # weight norm
                if attentionLayerParams.norm:
                        weights_norm = layer_norm(weights_logits,name="weights_norm")
                else:
                        weights_norm = weights_logits
                #mask
                if attentionLayerParams.zero_pad:
                        mask = tf.sign(tf.reduce_sum(tf.abs(inputs), axis=-1)) #N,SL
                        mask = tf.tile(tf.expand_dims(mask,1),[1,tf.shape(query)[1],1])#N,m,SL
                        paddings = tf.ones_like(weights_norm)*(-2**32+1)
                        weights_pad = tf.where(tf.equal(mask, 0), paddings, weights_norm)
                else:
                        weights_pad = weights_norm
 
                # normalzie
                weights =  tf.nn.softmax(weights_pad)
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

                return outputs,weights,weights_logits
def semantic_attention(inputs,query,scope_name):
        with tf.variable_scope(scope_name):
                #apply the layer normalization
                norm_inputs = tf.concat([inputs,tf.tile(query,[1,int(inputs.get_shape()[1]),1])],2)                
                # if baseParams.norm:
                #         norm_inputs = layer_norm(tf.concat([inputs,tf.tile(query,[1,int(inputs.get_shape()[1]),1])],2),name='inputs')
                # else:
                        
                #the keys and values output dim must be the same with the queries dim
                output_dim = int(query.get_shape()[2])
                # Linear projections
                K = tf.layers.dense(norm_inputs, output_dim, activation=tf.nn.tanh,name="key") # (N, SL, QD)
                V = tf.layers.dense(norm_inputs, output_dim, activation=tf.nn.tanh,name="value") # (N, SL, QD)
                Q = tf.layers.dense(query, output_dim, activation=tf.nn.tanh,name="query") # (N, 1, QD)
                # Multiplication
                weights_norm = tf.matmul(Q,tf.transpose(K,[0,2,1])) # (N,m,SL)
                # # weight norm
                # if baseParams.norm:
                #         weights_norm = layer_norm(weights_logits,name="weights_norm")
                # else:
                #          = weights_logits
                #mask
                if baseParams.zero_pad:
                        mask = tf.sign(tf.reduce_sum(tf.abs(inputs), axis=-1)) #N,SL
                        mask = tf.tile(tf.expand_dims(mask,1),[1,tf.shape(query)[1],1])#N,m,SL
                        paddings = tf.ones_like(weights_norm)*(-2**32+1)
                        weights_pad = tf.where(tf.equal(mask, 0), paddings, weights_norm)
                else:
                        weights_pad = weights_norm
                # normalzie
                weights =  tf.nn.softmax(weights_pad)
                # Weighted sum
                outputs = tf.matmul(weights, V) # (N,m,QD)

                if baseParams.direct_cont:
                        # Residual connection
                        outputs = tf.concat([outputs,query],2,name="outputs")
                        
                return outputs 
                
def semantic_attention(inputs,query,scope_name,is_training):
        with tf.variable_scope(scope_name):
                #apply the layer normalization                
                if attentionLayerParams.norm:
                        norm_inputs = layer_norm(tf.concat([inputs,tf.tile(query,[1,int(inputs.get_shape()[1]),1])],2),name='inputs')
                else:
                        norm_inputs = tf.concat([inputs,tf.tile(query,[1,int(inputs.get_shape()[1]),1])],2)
                #the keys and values output dim must be the same with the queries dim
                output_dim = int(query.get_shape()[2])
                # Linear projections
                K = tf.layers.dense(norm_inputs, output_dim, activation=activation_fn,name="key") # (N, SL, QD)
                V = tf.layers.dense(norm_inputs, output_dim, activation=activation_fn,name="value") # (N, SL, QD)
                Q = tf.layers.dense(query, output_dim, activation=activation_fn,name="query") # (N, 1, QD)
                # Multiplication
                weights_logits = tf.matmul(Q,tf.transpose(K,[0,2,1])) # (N,m,SL)
                # weight norm
                if attentionLayerParams.norm:
                        weights_norm = layer_norm(weights_logits,name="weights_norm")
                else:
                        weights_norm = weights_logits
                #mask
                if attentionLayerParams.zero_pad:
                        mask = tf.sign(tf.reduce_sum(tf.abs(inputs), axis=-1)) #N,SL
                        mask = tf.tile(tf.expand_dims(mask,1),[1,tf.shape(query)[1],1])#N,m,SL
                        paddings = tf.ones_like(weights_norm)*(-2**32+1)
                        weights_pad = tf.where(tf.equal(mask, 0), paddings, weights_norm)
                else:
                        weights_pad = weights_norm
                # normalzie
                weights =  tf.nn.softmax(weights_pad)
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

                return outputs,weights,weights_logits
 
def entity_attention(inputs,query,scope_name):
    # query    (QD,eN)
    # inputs   (N,TL,WD)
        with tf.variable_scope(scope_name):
                entity_dim = int(query.get_shape()[0])
                multi_query = tf.tile(tf.expand_dims(query,0),[tf.shape(inputs)[0],1,1])
                # Linear projections
                K = tf.layers.dense(inputs, entity_dim, activation=tf.nn.tanh,name="key") # (N, SL, QD)
                V = tf.layers.dense(inputs, entity_dim, activation=tf.nn.tanh,name="value") # (N, SL, QD)
                # Multiplication
                weights_norm = tf.matmul(K,multi_query) # (N,SL,eN)
          
                if baseParams.zero_pad:
                        mask = tf.sign(tf.reduce_sum(tf.abs(inputs), axis=-1)) #N,SL
                        mask = tf.tile(tf.expand_dims(mask,2),[1,1,tf.shape(query)[1]])#N,SL,eN
                        paddings = tf.ones_like(weights_norm)*(-2**32+1)
                        weights_pad = tf.where(tf.equal(mask, 0), paddings, weights_norm)
                else:
                        weights_pad = weights_norm
                # normalzie
                weights =  tf.nn.softmax(weights_pad,dim=1)
                # Weighted sum
                weights = tf.transpose(weights,[0,2,1]) # (N,eN,SL)
                outputs = tf.matmul(weights, V) # (N,eN,QD)

                return tf.reshape(outputs,[-1,entity_dim*baseParams.entity_num])
