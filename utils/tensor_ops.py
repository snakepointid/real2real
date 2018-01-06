# -*- coding: utf-8 -*-
#/usr/bin/python2
 

from __future__ import print_function
import tensorflow as tf
from real2real.app.params import embedLayerParams,baseLayerParams
import math
from pydoc import locate

def noam_norm(x, epsilon=1.0, name=None):
        """One version of layer normalization."""
        with tf.name_scope(name, default_name="noam_norm", values=[x]):
                shape = x.get_shape()
                ndims = len(shape)
                result = tf.nn.l2_normalize(x, ndims - 1, epsilon=epsilon)
                return result 

def cos_similar(x,y,k=None,transpose=True):
        x_norm = noam_norm(x,name='x')
        y_norm = noam_norm(y,name='y')
        result = tf.matmul(x_norm,y_norm,transpose_b=transpose,name="cosine_similarity")	
        if k:
        	_,result = tf.nn.top_k(result,k)
        return result

def dot_similar(x,y,k=None,transpose=True):
        result = tf.matmul(x,y,transpose_b=transpose,name="product_similarity")
        if k:
        	_,result = tf.nn.top_k(result,k)
        return result

def distance_similar(x,y,k=None):
	square_diff = tf.square(tf.expand_dims(x,1)-tf.expand_dims(y,0))
	result = tf.sqrt(tf.reduce_mean(square_diff,-1))
	if k:
                _,result = tf.nn.top_k(-result,k)
	return result
def compute_similariry(x,y,k,types='cos'):
        if types=='cos':
                return cos_similar(x,y,k)
        elif types=='dot':
                return dot_similar(x,y,k)
        else:
                return distance_similar(x,y,k)