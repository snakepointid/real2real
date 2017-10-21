from __future__ import print_function
import tensorflow as tf
from real2real.layers.common_layers import *
from pydoc import locate
from real2real.app.params import convLayerParams 
 
def multiLayer_conv_strip(inputs,multi_cnn_params,scope_name,is_training,is_dropout,reuse=None):
        
        if len(inputs.get_shape())!=3:
                return inputs

        kernel_size,stride_step,conv_layer_num = multi_cnn_params
        activation_fn = locate(convLayerParams.activation_fn)
        with tf.variable_scope(scope_name,reuse=reuse):
                next_input = tf.contrib.layers.dropout(
                                        inputs=inputs,
                                        keep_prob=convLayerParams.dropout_rate,
                                        is_training=is_dropout) 
                for layer_idx in range(conv_layer_num):
                        cnn_out = strip_conv(
                                        inputs=next_input,
                                        kernel_size=kernel_size,
                                        stride_step=stride_step,
                                        scope_name="conv_layer{}".format(layer_idx),
                                        is_training = is_training)
                        next_input = activation_fn(cnn_out)
                        
        return  next_input 

def strip_conv(inputs,kernel_size,stride_step,scope_name,is_training):
        static_shape  = inputs.get_shape()
        with tf.variable_scope(scope_name,reuse=None):
                filter_kernels = tf.get_variable('kernel', shape=[kernel_size,static_shape[2],convLayerParams.filter_nums],trainable=is_training)
                cnn_kernel = tf.nn.conv1d(inputs, filter_kernels, stride= stride_step, padding='VALID')
                cnn_bias = tf.get_variable('bias',shape=(convLayerParams.filter_nums,), initializer=tf.constant_initializer(0),trainable=is_training)
                cnn_output = cnn_kernel+cnn_bias

                if convLayerParams.norm:        
                        cnn_output = tf.layers.batch_normalization(cnn_output)  

                if convLayerParams.zero_pad:
                        cnn_mask = tf.sign(tf.reduce_sum(tf.abs(cnn_kernel), axis=-1)) #N,SL
                        cnn_mask = tf.tile(tf.expand_dims(cnn_mask,2),[1,1,convLayerParams.filter_nums])#N,SL,FN
                        paddings = tf.zeros_like(cnn_output) 
                        cnn_output = tf.where(tf.equal(cnn_mask, 0), paddings, cnn_output) # (h*N, T_q, T_k)
        return cnn_output
 
def conv1d_to_full_layer(inputs,scope_name,is_training,reuse=None):
        static_shape  = inputs.get_shape()
        with tf.variable_scope(scope_name,reuse=reuse):
                filter_kernels = tf.get_variable('kernel', shape=[static_shape[1],static_shape[2],convLayerParams.filter_nums],trainable=is_training)
                cnn_bias      = tf.get_variable('bias'  , shape=(convLayerParams.filter_nums,), initializer=tf.constant_initializer(0),trainable=is_training)
                cnn_output  = tf.nn.conv1d(inputs, filter_kernels,stride=1, padding='VALID')+ cnn_bias
        return tf.squeeze(cnn_output,1) 

def conv2d_to_full_layer(inputs,scope_name,is_training,reuse=None):
        static_shape  = inputs.get_shape()
        with tf.variable_scope(scope_name,reuse=reuse):
                filter_kernels = tf.get_variable('kernel', shape=[static_shape[1],static_shape[2],static_shape[3],convLayerParams.filter_nums],trainable=is_training)
                cnn_bias      = tf.get_variable('bias'  , shape=(convLayerParams.filter_nums,), initializer=tf.constant_initializer(0),trainable=is_training)
                cnn_output  = tf.nn.conv1d(inputs, filter_kernels,stride=1, padding='VALID')+ cnn_bias
        return tf.squeeze(cnn_output,[1,2])
