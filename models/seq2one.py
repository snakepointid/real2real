from __future__ import print_function
import tensorflow as tf

from real2real.models.base_model import regressModel,multiClsModel

from real2real.modules.text_encoder import sentence_encoder
from real2real.modules.full_connector import final_mlp_encoder
from real2real.modules.entity_encoder import *

from real2real.utils.shape_ops import *

from real2real.app.params import ctrRankModelParams,newsClsModelParams
   
class NewsClsModel(multiClsModel):
            def _build_(self):
                        # input coding placeholder
                        self.title_source = tf.placeholder(shape=(None, newsClsModelParams.title_maxlen),dtype=tf.int64)
                        self.content_source = tf.placeholder(shape=(None, newsClsModelParams.content_maxlen),dtype=tf.int64)
                        self.target = tf.placeholder(shape=(None, ),dtype=tf.int32)
                        #target to token embedding
                        token_context = tf.get_variable('token_context',
                                                       dtype=tf.float32,
                                                       shape=[newsClsModelParams.target_vocab_size, newsClsModelParams.embedding_dim],
                                                       initializer=tf.contrib.layers.xavier_initializer(),
                                                       trainable=self.is_training)
                        #title encoding
                        title_encoding = sentence_encoder(
                                                       inputs=self.title_source,
                                                       query=token_context,
                                                       vocab_size=newsClsModelParams.source_vocab_size,
                                                       multi_cnn_params=newsClsModelParams.token_cnn_params,#kernel,stride,layer
                                                       scope='sentence',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=None) #N,FN
                         
                        content_encoding = sentence_encoder(
                                                       inputs=self.content_source,
                                                       query=token_context,
                                                       vocab_size=newsClsModelParams.source_vocab_size,
                                                       multi_cnn_params=newsClsModelParams.token_cnn_params,#kernel,stride,layer
                                                       scope='sentence',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=True)#N,FN

                        full_layer = tf.concat([content_encoding,title_encoding],1)
                                    
                        self.logits = final_mlp_encoder(
                                             inputs=full_layer,
                                             output_dim=newsClsModelParams.target_vocab_size,
                                             is_training=self.is_training,
                                             is_dropout=self.is_dropout) 

class CtrRankModel(regressModel):
            def _build_(self):
                        # input coding placeholder
                        self.title_source = tf.placeholder(shape=(None, ctrRankModelParams.title_maxlen),dtype=tf.int64)
                        self.recall_tag = tf.placeholder(shape=(None,),dtype=tf.int64)
                        self.target = tf.placeholder(shape=(None, ),dtype=tf.int32)
                        #target to token embedding
                        tag_embed = tag_embedding(
                                                inputs=tf.reshape(self.recall_tag,[-1,1]),
                                                vocab_size=ctrRankModelParams.tag_size,
                                                is_training=self.is_training,
                                                scope='recalltag')
                        #title encoding
                        title_encoding = sentence_encoder(
                                                       inputs=self.title_source,
                                                       query=tag_embed,
                                                       vocab_size=ctrRankModelParams.source_vocab_size,
                                                       multi_cnn_params=ctrRankModelParams.token_cnn_params,#kernel,stride,layer
                                                       scope='sentence',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=None) #N,1,FN
                                    
                        self.logits = final_mlp_encoder(
                                             inputs=full_layer,
                                             output_dim=ctrRankModelParams.target_vocab_size,
                                             is_training=self.is_training,
                                             is_dropout=self.is_dropout)                                               

