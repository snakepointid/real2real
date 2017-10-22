# -*- coding: utf-8 -*-
#/usr/bin/python2
 

from __future__ import print_function
import tensorflow as tf
from real2real.app.params import embedLayerParams,baseLayerParams
import math
from pydoc import locate

def split_long_text(inputs,short_length):
        static_shape = inputs.get_shape()
        splited_num = int(int(static_shape[1])/short_length)
        outputs = tf.reshape(inputs,[-1,short_length])
        return outputs,splited_num

def stack_short_encode(inputs,splited_num):
	static_shape = inputs.get_shape()
        outputs = tf.reshape(inputs,[-1,splited_num,int(static_shape[1])])
        return outputs

def label_smoothing(inputs, epsilon=0.1):
        K = inputs.get_shape().as_list()[-1] # number of channels
        return ((1-epsilon) * inputs) + (epsilon / K)
