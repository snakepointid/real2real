from __future__ import print_function
import tensorflow as tf

from real2real.models.base_model import regressModel,multiClsModel

from real2real.modules.text_encoder import text_conv_encoder,text_atten_encoder
from real2real.modules.full_connector import final_mlp_encoder

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
                                                       multi_cnn_params=ctrRankModelParams.token_cnn_params,#kernel,stride,layers                                                                                                             
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
                        self.logits = final_mlp_encoder(
                                                         inputs=full_layer,
                                                         output_dim=1,
                                                         is_training=self.is_training,
                                                         is_dropout=self.is_dropout)

class StackAttenCls(multiClsModel):
            def _build_(self):
                        # input coding placeholder
                        self.title_source = tf.placeholder(shape=(None, newsClsModelParams.title_maxlen),dtype=tf.int64)
                        self.content_source = tf.placeholder(shape=(None, newsClsModelParams.content_maxlen),dtype=tf.int64)
                        self.target = tf.placeholder(shape=(None, ),dtype=tf.int32)
                        #target to token embedding
                        token_context = tf.get_variable('token_context',
                                                       dtype=tf.float32,
                                                       shape=[newsClsModelParams.target_vocab_size, embedLayerParams.embedding_dim],
                                                       initializer=tf.contrib.layers.xavier_initializer(),
                                                       trainable=self.is_training)
                        #title encoding
                        title_encoding = text_atten_encoder(
                                                       inputs=self.title_source,
                                                       query=token_context,
                                                       vocab_size=newsClsModelParams.source_vocab_size,
                                                       multi_cnn_params=newsClsModelParams.token_cnn_params,#kernel,stride,layer
                                                       scope='sentence',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=None) #N,FN
                        #content encoding
                        split_content,sentence_num = split_long_text(self.content_source,newsClsModelParams.title_maxlen)

                        content_encoding = text_atten_encoder(
                                                       inputs=split_content,
                                                       query=token_context,
                                                       vocab_size=newsClsModelParams.source_vocab_size,
                                                       multi_cnn_params=newsClsModelParams.token_cnn_params,#kernel,stride,layer
                                                       scope='sentence',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=True)#N*ST,FN
                        
                        stack_content = stack_short_encode(content_encoding,sentence_num)#N,ST,FN
                        #target to sentence embedding
                        sentence_context = tf.get_variable('sentence_context',
                                                       dtype=tf.float32,
                                                       shape=[newsClsModelParams.target_vocab_size, embedLayerParams.embedding_dim],
                                                       initializer=tf.contrib.layers.xavier_initializer(),
                                                       trainable=self.is_training)

                        content_encoding = text_atten_encoder(
                                                       inputs=stack_content,
                                                       query=sentence_context,
                                                       vocab_size=newsClsModelParams.source_vocab_size,
                                                       multi_cnn_params=newsClsModelParams.sentence_cnn_params,#kernel,stride,layer
                                                       scope='doc',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=None) ##N,FN      
                        #full_layer
                        if newsClsModelParams.mode == 'content':
                                    full_layer = content_encoding 
                        elif newsClsModelParams.mode == 'title':    
                                    full_layer = title_encoding      
 
                        self.logits = final_mlp_encoder(
                                             inputs=full_layer,
                                             output_dim=newsClsModelParams.target_vocab_size,
                                             is_training=self.is_training,
                                             is_dropout=self.is_dropout)
   
class DirectAttenCls(multiClsModel):
            def _build_(self):
                        # input coding placeholder
                        self.title_source = tf.placeholder(shape=(None, newsClsModelParams.title_maxlen),dtype=tf.int64)
                        self.content_source = tf.placeholder(shape=(None, newsClsModelParams.content_maxlen),dtype=tf.int64)
                        self.target = tf.placeholder(shape=(None, ),dtype=tf.int32)
                        #target to token embedding
                        token_context = tf.get_variable('token_context',
                                                       dtype=tf.float32,
                                                       shape=[newsClsModelParams.target_vocab_size, embedLayerParams.embedding_dim],
                                                       initializer=tf.contrib.layers.xavier_initializer(),
                                                       trainable=self.is_training)
                        #title encoding
                        title_encoding = text_atten_encoder(
                                                       inputs=self.title_source,
                                                       query=token_context,
                                                       vocab_size=newsClsModelParams.source_vocab_size,
                                                       multi_cnn_params=newsClsModelParams.token_cnn_params,#kernel,stride,layer
                                                       scope='sentence',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=None) #N,FN
                         
                        content_encoding = text_atten_encoder(
                                                       inputs=self.content_source,
                                                       query=token_context,
                                                       vocab_size=newsClsModelParams.source_vocab_size,
                                                       multi_cnn_params=newsClsModelParams.token_cnn_params,#kernel,stride,layer
                                                       scope='sentence',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=True)#N,FN

                        #full_layer
                        if newsClsModelParams.mode == 'content':
                                    full_layer = content_encoding 
                        elif newsClsModelParams.mode == 'title':    
                                    full_layer = title_encoding    

                        self.logits = final_mlp_encoder(
                                             inputs=full_layer,
                                             output_dim=newsClsModelParams.target_vocab_size,
                                             is_training=self.is_training,
                                             is_dropout=self.is_dropout)  

