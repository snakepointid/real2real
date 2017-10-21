from __future__ import print_function
import tensorflow as tf

from real2real.models.base_model import regressModel,multiClsModel

from real2real.modules.text_encoder import text_conv_encoder,text_atten_encoder
from real2real.modules.full_connector import multi_layer_perceptron

from real2real.utils.shape_ops import *

from real2real.app.params import ctrRankModelParams,newsClsModelParams,embedLayerParams
 
class ConvRank(regressModel):
            def _build_(self):
                        # input coding placeholder
                        self.source = tf.placeholder(shape=(None, ctrRankModelParams.source_maxlen),dtype=tf.int64)
                        self.tag = tf.placeholder(shape=(None, ),dtype=tf.int64)
                        self.target = tf.placeholder(shape=(None, ),dtype=tf.float32)

                        encoding = text_conv_encoder(
                                                       inputs=self.source,
                                                       vocab_size=ctrRankModelParams.source_vocab_size,                                                     
                                                       multi_cnn_params=[5,2,3],#kernel,stride,layers                                                                                                             
                                                       maxlen=ctrRankModelParams.source_maxlen,
                                                       scope='title',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=None)
                        tag_embed = embedding(
                                          inputs=self.tag,
                                          vocab_size=ctrRankModelParams.tag_size,                                         
                                          zero_pad=False,                                         
                                          scope="tag_embed")
                        #forward feed connect
                        full_layer = tf.concat([tag_embed,encoding],1)
                        self.logits = multi_layer_perceptron(
                                                         inputs=full_layer,
                                                         output_dim=1,
                                                         is_training=self.is_training,
                                                         is_dropout=self.is_dropout)

 
class ConvCls(multiClsModel):
            def _build_(self):
                        # input coding placeholder
                        self.title_source = tf.placeholder(shape=(None, newsClsModelParams.title_maxlen),dtype=tf.int64)
                        self.content_source = tf.placeholder(shape=(None, newsClsModelParams.content_maxlen),dtype=tf.int64)
                        self.target = tf.placeholder(shape=(None, ),dtype=tf.int32)
                        
                        title_encoding = text_conv_encoder(
                                                       inputs=self.title_source,
                                                       vocab_size=newsClsModelParams.source_vocab_size,
                                                       multi_cnn_params=[5,2,3],#kernel,stride,layers
                                                       maxlen=newsClsModelParams.title_maxlen,
                                                       scope='sentence',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=None) #N,FN

                        split_content,sentence_num = split_long_text(self.content_source,newsClsModelParams.title_maxlen)
                        content_encoding = text_conv_encoder(
                                                       inputs=split_content,
                                                       vocab_size=newsClsModelParams.source_vocab_size,
                                                       multi_cnn_params=[5,2,3],#kernel,stride,layers
                                                       maxlen=newsClsModelParams.title_maxlen,
                                                       scope='sentence',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=True)   #N*ST,FN

                        stack_content = stack_short_encode(content_encoding,sentence_num)#N,ST,FN

                        content_encoding = text_conv_encoder(
                                                       inputs=stack_content,
                                                       vocab_size=newsClsModelParams.source_vocab_size,
                                                       multi_cnn_params=[3,1,1],#kernel,stride,layer
                                                       maxlen=newsClsModelParams.content_maxlen,
                                                       scope='doc',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=None)   #N,FN
                        
                        #full_layer = tf.concat([title_out,content_out],1)
                        #full_layer = content_encoding
                        full_layer = content_encoding
                        self.logits = multi_layer_perceptron(
                                                         inputs=full_layer,
                                                         output_dim=newsClsModelParams.target_vocab_size,
                                                         is_training=self.is_training,
                                                         is_dropout=self.is_dropout)
class AttenCls(multiClsModel):
            def _build_(self):
                        # input coding placeholder
                        self.title_source = tf.placeholder(shape=(None, newsClsModelParams.title_maxlen),dtype=tf.int64)
                        self.content_source = tf.placeholder(shape=(None, newsClsModelParams.content_maxlen),dtype=tf.int64)
                        self.target = tf.placeholder(shape=(None, ),dtype=tf.int32)
                        #target to token embedding
                        target_token_embed = tf.get_variable('target_token_embed',
                                                       dtype=tf.float32,
                                                       shape=[newsClsModelParams.target_vocab_size, embedLayerParams.embedding_dim],
                                                       initializer=tf.contrib.layers.xavier_initializer(),
                                                       trainable=self.is_training)
                        #title encoding
                        title_encoding = text_atten_encoder(
                                                       inputs=self.title_source,
                                                       query=target_token_embed,
                                                       vocab_size=newsClsModelParams.source_vocab_size,
                                                       multi_cnn_params=[3,1,1],#kernel,stride,layer
                                                       maxlen=newsClsModelParams.title_maxlen,
                                                       scope='sentence',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=None) #N,m,FN
                        #content encoding
                        split_content,sentence_num = split_long_text(self.content_source,newsClsModelParams.title_maxlen)
                        content_encoding = text_atten_encoder(
                                                       inputs=split_content,
                                                       query=target_token_embed,
                                                       vocab_size=newsClsModelParams.source_vocab_size,
                                                       multi_cnn_params=[3,1,1],#kernel,stride,layer
                                                       maxlen=newsClsModelParams.title_maxlen,
                                                       scope='sentence',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=True)   #N*ST,m,FN
                        
			stack_content = stack_short_encode(content_encoding,sentence_num)#N,ST,m,FN
                        #target to sentence embedding
                        target_sentence_embed = tf.get_variable('target_sentence_embed',
                                                       dtype=tf.float32,
                                                       shape=[newsClsModelParams.target_vocab_size, embedLayerParams.embedding_dim],
                                                       initializer=tf.contrib.layers.xavier_initializer(),
                                                       trainable=self.is_training)

                        content_encoding = text_atten_encoder(
                                                       inputs=stack_content,
                                                       query=target_sentence_embed,
                                                       vocab_size=newsClsModelParams.source_vocab_size,
                                                       multi_cnn_params=[1,1,1],#kernel,stride,layer
                                                       maxlen=newsClsModelParams.content_maxlen,
                                                       scope='doc',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=None)   ##N*m,m,FN			 
                        #full connect
                        self.logits = multi_layer_perceptron(
                                                         inputs=content_encoding,
                                                         output_dim=1,
                                                         is_training=self.is_training,
                                                         is_dropout=self.is_dropout)

