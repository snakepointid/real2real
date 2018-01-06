from __future__ import print_function
import tensorflow as tf
from real2real.layers.common_layers import *
from pydoc import locate


def max_pool_layer(inputs):
	#apply the simple max ops
        return tf.reduce_max(inputs,1)

def avg_pool_layer(inputs):
	#apply the simple max ops
        sums = tf.reduce_sum(inputs,1)#N,D
        seq_length = tf.reduce_sum(tf.sign(tf.reduce_sum(tf.abs(inputs), axis=-1)),1,keep_dims=True)#N
        seq_length = tf.tile(seq_length,[1,tf.shape(sums)[1]])
        return tf.expand_dims(sums/seq_length,1)
