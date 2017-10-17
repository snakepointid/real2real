from __future__ import print_function
import tensorflow as tf
from real2real.layers.common_layers import *
from pydoc import locate
from real2real.app.params import convLayerParams 
 
def multiLayer_conv_strip(inputs,kernel_size,conv_layer_num,scope_name,is_training,is_dropout):
        activation_fn = locate(convLayerParams.activation_fn)
        with tf.variable_scope(scope_name):
                # Apply dropout to embeddings
                inputs = tf.contrib.layers.dropout(
                                        inputs=inputs,
                                        keep_prob=convLayerParams.dropout_rate,
                                        is_training=is_dropout)

                cnn_a_output = inputs
                for layer_idx in range(conv_layer_num):
                        next_layer = strip_conv(
                                              inputs=cnn_a_output,
                                              kernel_size=kernel_size,
                                              scope_name="conv_layer{}".format(layer_idx),
                                              is_training = is_training)
                        next_layer = tf.layers.batch_normalization(next_layer)          
                        cnn_a_output = activation_fn(next_layer)
        return	cnn_a_output 

def strip_conv(inputs,kernel_size,scope_name,is_training):
        static_shape  = inputs.get_shape()
        with tf.variable_scope(scope_name,reuse=None):
                filter_kernels = tf.get_variable('kernel', shape=[kernel_size,static_shape[2],convLayerParams.filter_nums],trainable=is_training)
                cnn_bias      = tf.get_variable('bias'  , shape=(convLayerParams.filter_nums,), initializer=tf.constant_initializer(0),trainable=is_training)
                cnn_output  = tf.nn.conv1d(inputs, filter_kernels, stride= convLayerParams.strip_step, padding='VALID')+ cnn_bias
        return cnn_output

def conv_to_full_layer(inputs,scope,is_training):
	activation_fn = locate(convLayerParams.activation_fn)
	static_shape  = inputs.get_shape()
	if len(static_shape)==3:
		outputs = conv1d_to_full_layer(inputs,scope,is_training)
	elif len(static_shape)==4:
		outputs = conv2d_to_full_layer(inputs,scope,is_training)
	else:
		raise ValueError("input shape's rank  must be 3 or 4")
	return activation_fn(outputs)

def conv1d_to_full_layer(inputs,scope_name,is_training):
	static_shape  = inputs.get_shape()
	with tf.variable_scope(scope_name,reuse=None):
                filter_kernels = tf.get_variable('kernel', shape=[static_shape[1],static_shape[2],convLayerParams.filter_nums],trainable=is_training)
                cnn_bias      = tf.get_variable('bias'  , shape=(convLayerParams.filter_nums,), initializer=tf.constant_initializer(0),trainable=is_training)
                cnn_output  = tf.nn.conv1d(inputs, filter_kernels, stride= convLayerParams.strip_step, padding='VALID')+ cnn_bias
        return tf.squeeze(cnn_output,1) 

def conv2d_to_full_layer(inputs,scope_name,is_training):
        static_shape  = inputs.get_shape()
        with tf.variable_scope(scope_name,reuse=None):
                filter_kernels = tf.get_variable('kernel', shape=[static_shape[1],static_shape[2],static_shape[3],convLayerParams.filter_nums],trainable=is_training)
                cnn_bias      = tf.get_variable('bias'  , shape=(convLayerParams.filter_nums,), initializer=tf.constant_initializer(0),trainable=is_training)
                cnn_output  = tf.nn.conv1d(inputs, filter_kernels, stride= convLayerParams.strip_step, padding='VALID')+ cnn_bias
        return tf.squeeze(cnn_output,[1,2])
