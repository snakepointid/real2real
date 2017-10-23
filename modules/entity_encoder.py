# -*- coding: utf-8 -*-
import tensorflow as tf

from real2real.layers.common_layers import embedding,positional_encoding
from real2real.app.params import entityEmbedModuleParams
from pydoc import locate

def semantic_position_embedding(inputs,vocab_size,is_training,scope,reuse=None):
        with tf.variable_scope(scope,reuse=reuse):
                encoding = embedding(inputs=inputs, 
                                     vocab_size=vocab_size,                                
                                     zero_pad=entityEmbedModuleParams.zero_pad,
                                     reuse=reuse,
                                     is_training=is_training,
                                     scope="token")

                if not entityEmbedModuleParams.flag_position_embed:
                        return encoding
                ## Positional Encoding
                position_code = tf.tile(tf.expand_dims(tf.to_int64(tf.range(tf.shape(inputs)[1])+1), 0), [tf.shape(inputs)[0], 1])
                paddings = tf.zeros_like(inputs)
                position_code = tf.where(tf.equal(inputs, 0), paddings, position_code)
                #position embed
                maxlen = int(inputs.get_shape()[-1])
                if entityEmbedModuleParams.flag_sinusoid:
                        encoding += positional_encoding(
                                        inputs=position_code,
                                        vocab_size=maxlen+1,  
                                        scope="position")
                else:
                        encoding += embedding(
                                        inputs=position_code,
                                        vocab_size=maxlen+1, 
                                        zero_pad=entityEmbedModuleParams.zero_pad, 
                                        reuse=reuse,
                                        is_training=is_training,
                                        scope="position")
        return encoding


def tag_embedding(inputs,vocab_size,is_training,scope,reuse=None):
        with tf.variable_scope(scope,reuse=reuse):
                encoding = embedding(inputs=inputs, 
                                     vocab_size=vocab_size,                                
                                     zero_pad=False,
                                     reuse=reuse,
                                     is_training=is_training,
                                     scope="tag")
        return encoding
