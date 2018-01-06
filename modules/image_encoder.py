# -*- coding: utf-8 -*-
#/usr/bin/python2

import tensorflow as tf

from real2real.layers.conv_layers import strip_conv,max_pool_layer
from real2real.layers.attention_layers import target_attention
'''
    this is a general text encoding module
    it can be consisted by three parts: conv,attention and pool layers
'''
def image_encoder(inputs,query,layers,multi_cnn_params,scope,is_training,is_dropout,reuse):
        #apply the dropout to the inputs
        encoding = tf.contrib.layers.dropout(
                                           inputs=inputs,
                                           keep_prob=0.5,
                                           is_training=is_dropout) 

        with tf.variable_scope(scope,reuse=reuse):              
                #multi layers stride convolution without pool
                if 'C' in layers:
                        kernel_size,stride_step,conv_layer_num = multi_cnn_params
                        with tf.variable_scope('ml_cov',reuse=reuse):
                                for layer_idx in range(conv_layer_num):
                                        encoding = strip_conv(
                                                       inputs=encoding,
                                                       kernel_size=kernel_size,
                                                       stride_step=stride_step,
                                                       scope_name="conv_layer{}".format(layer_idx),
                                                       is_training = is_training,
                                                       is_dropout=is_dropout)
                 
        return encoding

 
 
# Variables for weights and biases
with tf.variable_scope('encoding'):
    # After converting the input to a square image, we apply the first convolution, using 2x2 kernels
    with tf.variable_scope('conv1'):
        wec1 = tf.get_variable('w', shape=(5, 5, 1, m_c1), initializer=tf.truncated_normal_initializer())
        bec1 = tf.get_variable('b', shape=(m_c1,), initializer=tf.constant_initializer(0))
    # Second convolution
    with tf.variable_scope('conv2'):
        wec2 = tf.get_variable('w', shape=(3, 3, m_c1, m_c2), initializer=tf.truncated_normal_initializer())
        bec2 = tf.get_variable('b', shape=(m_c2,), initializer=tf.constant_initializer(0))
    # First fully connected layer
    with tf.variable_scope('fc1'):
        wef1 = tf.get_variable('w', shape=(5*5*m_c2, n_h1), initializer=tf.contrib.layers.xavier_initializer())
        bef1 = tf.get_variable('b', shape=(n_h1,), initializer=tf.constant_initializer(0))
    # Second fully connected layer
    with tf.variable_scope('fc2'):
        wef2 = tf.get_variable('w', shape=(n_h1, n_h2), initializer=tf.contrib.layers.xavier_initializer())
        bef2 = tf.get_variable('b', shape=(n_h2,), initializer=tf.constant_initializer(0))

reshaped_x = tf.reshape(x, (-1, 28, 28, 1))
y1 = tf.nn.conv2d(reshaped_x, wec1, strides=(1, 2, 2, 1), padding='VALID')
#y2 = tf.nn.elu(y1 + bec1)
y2 = tf.nn.elu(y1)
y3 = tf.nn.conv2d(y2, wec2, strides=(1, 2, 2, 1), padding='VALID')
#y4 = tf.nn.elu(y3 + bec2)
y4 = tf.nn.elu(y3 )
y5 = tf.reshape(y4, (-1, 5*5*m_c2))
y6 = tf.nn.elu(tf.matmul(y5, wef1) + bef1)
encode = tf.nn.elu(tf.matmul(y6, wef2) + bef2)

with tf.variable_scope('decoding'):
    # for the transposed convolutions, we use the same weights defined above
    with tf.variable_scope('fc1'):
        wdf1 = tf.get_variable('w', shape=(n_h2,n_h1), initializer=tf.contrib.layers.xavier_initializer())
        bdf1 = tf.get_variable('b', shape=(n_h1,), initializer=tf.constant_initializer(0))
    with tf.variable_scope('fc2'):
        wdf2 = tf.get_variable('w', shape=(n_h1,5*5*m_c2), initializer=tf.contrib.layers.xavier_initializer())
        bdf2 = tf.get_variable('b', shape=(5*5*m_c2,), initializer=tf.constant_initializer(0))
    with tf.variable_scope('deconv1'):
        wdd1 = tf.get_variable('w', shape=(3, 3, m_c1, m_c2), initializer=tf.contrib.layers.xavier_initializer())
        bdd1 = tf.get_variable('b', shape=(m_c1,), initializer=tf.constant_initializer(0))
    with tf.variable_scope('deconv2'):
        wdd2 = tf.get_variable('w', shape=(5, 5, 1, m_c1), initializer=tf.contrib.layers.xavier_initializer())
        bdd2 = tf.get_variable('b', shape=(1,), initializer=tf.constant_initializer(0))

u1 = tf.nn.elu(tf.matmul(encode, wdf1) + bdf1)
u2 = tf.nn.elu(tf.matmul(u1, wdf2) + bdf2)
u3 = tf.reshape(u2, tf.shape(y3))
#u4 = tf.nn.conv2d_transpose(u3, wdd1, output_shape=tf.shape(y1), strides=(1, 2, 2, 1), padding='VALID')
u4 = tf.nn.conv2d_transpose(u3, wec2, output_shape=tf.shape(y1), strides=(1, 2, 2, 1), padding='VALID')
#u5 = tf.nn.elu(u4 + bdd1)
u5 = tf.nn.elu(u4)
#u6 = tf.nn.conv2d_transpose(u5, wdd2, output_shape=(batch_size, 28, 28, 1), strides=(1, 2, 2, 1), padding='VALID')
u6 = tf.nn.conv2d_transpose(u5, wec1, output_shape=(batch_size, 28, 28, 1), strides=(1, 2, 2, 1), padding='VALID')
u7 = tf.nn.elu(u6)
#u7 = tf.nn.elu(u6 + bdd2)
decode = tf.reshape(u7, (-1, 784))

loss = tf.reduce_mean(tf.square(y - decode))
opt = tf.train.AdamOptimizer(0.001).minimize(loss)


