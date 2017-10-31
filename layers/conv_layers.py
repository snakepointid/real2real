from __future__ import print_function
import tensorflow as tf
from real2real.layers.common_layers import *
from pydoc import locate
from real2real.app.params import convLayerParams 
#load activate function
activation_fn = locate(convLayerParams.activation_fn) 
def strip_conv(inputs,kernel_size,stride_step,scope_name,is_training,is_dropout):
	#get the static shape of the input tensor
        static_shape  = inputs.get_shape()
        with tf.variable_scope(scope_name,reuse=None):
		#initial the convolutional kernel and bias
                filter_kernels = tf.get_variable('kernel', shape=[kernel_size,static_shape[2],convLayerParams.filter_nums],trainable=is_training)        
                cnn_bias = tf.get_variable('bias',shape=(convLayerParams.filter_nums,), initializer=tf.constant_initializer(0),trainable=is_training)
	#doing the conv process
        cnn_kernel = tf.nn.conv1d(inputs, filter_kernels, stride= stride_step, padding='VALID')
        cnn_output = cnn_kernel+cnn_bias
	#mask the zero pad
        if convLayerParams.zero_pad:
                cnn_mask = tf.sign(tf.reduce_sum(tf.abs(cnn_kernel), axis=-1)) #N,SL
                cnn_mask = tf.tile(tf.expand_dims(cnn_mask,2),[1,1,convLayerParams.filter_nums])#N,SL,FN
                paddings = tf.zeros_like(cnn_output) 
                cnn_output = tf.where(tf.equal(cnn_mask, 0), paddings, cnn_output) # (h*N, T_q, T_k)
	#apply the batch normalization
        if convLayerParams.norm:        
                cnn_output = tf.layers.batch_normalization(cnn_output)
	#activate the outputs
        cnn_output =  activation_fn(cnn_output) 
        return  cnn_output

def max_pool_layer(inputs):
	#apply the simple max ops
        return tf.reduce_max(inputs,1)
        
def conv_to_full_layer(inputs,scope_name,is_training,is_dropout):  
	#get the static shape of the input tensor
        static_shape  = inputs.get_shape()

        with tf.variable_scope(scope_name,reuse=None):
		#initial the convolutional kernel and bias
                filter_kernels = tf.get_variable('kernel', shape=[static_shape[1],static_shape[2],convLayerParams.filter_nums],trainable=is_training)
                cnn_bias      = tf.get_variable('bias'  , shape=(convLayerParams.filter_nums,), initializer=tf.constant_initializer(0),trainable=is_training)
	#doing the conv process
        cnn_output  = tf.nn.conv1d(inputs, filter_kernels,stride=1, padding='VALID')+ cnn_bias
        #activate the outputs
	cnn_output =  activation_fn(cnn_output) 
        return tf.reshape(cnn_output,[-1,convLayerParams.filter_nums])
 
