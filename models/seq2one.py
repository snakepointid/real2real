from __future__ import print_function
import tensorflow as tf

from real2real.models.base_model import regressModel,multiClsModel

from real2real.modules.text_encoder import *
from real2real.modules.full_connector import final_mlp_encoder
from real2real.modules.entity_encoder import *


from real2real.utils.shape_ops import *

from real2real.app.params import ctrRankModelParams,newsClsModelParams,keywordModelParams
      
      
class NewsClsModel(multiClsModel):
      def _build_(self):
      # input coding placeholder
            self.title_source = tf.placeholder(shape=(None, newsClsModelParams.title_maxlen),dtype=tf.int64)
            self.content_source = tf.placeholder(shape=(None, newsClsModelParams.content_maxlen),dtype=tf.int64)
            self.target = tf.placeholder(shape=(None, ),dtype=tf.int32)
            #target to token embedding
            token_context = tf.get_variable('token_context',
                                    dtype=tf.float32,
                                    shape=[newsClsModelParams.target_label_num, newsClsModelParams.embedding_dim],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    trainable=self.is_training)
            #embedding
            title_embed = semantic_position_embedding(
                                    inputs=self.title_source,
                                    vocab_size=newsClsModelParams.source_vocab_size,
                                    is_training=self.is_training,
                                    reuse=None,
                                    scope='chinese')
            #embedding
            content_embed = semantic_position_embedding(
                                    inputs=self.content_source,
                                    vocab_size=newsClsModelParams.source_vocab_size,
                                    is_training=self.is_training,
                                    reuse=True,
                                    scope='chinese')
            #title encoding
            title_encoding = sentence_encoder(
                                    inputs=title_embed,
                                    query=token_context,                                    
                                    multi_cnn_params=newsClsModelParams.title_cnn_params,#kernel,stride,layer
                                    scope='token',
                                    is_training=self.is_training,
                                    is_dropout=self.is_dropout,
                                    reuse=None) #N,FN
            content_encoding = sentence_encoder(
                                    inputs=content_embed,
                                    query=token_context, 
                                    multi_cnn_params=newsClsModelParams.title_cnn_params,#kernel,stride,layer
                                    scope='token',
                                    is_training=self.is_training,
                                    is_dropout=self.is_dropout,
                                    reuse=True)#N,FN
                     
            if newsClsModelParams.final_layer == "title":
                  full_layer = title_encoding
            elif newsClsModelParams.final_layer == "content":
                  full_layer = content_encoding  
            else:                        
                  full_layer = tf.concat([content_encoding,title_encoding],-1)
                                    
            self.logits = final_mlp_encoder(
                                    inputs=full_layer,
                                    output_dim=newsClsModelParams.target_label_num,
                                    is_training=self.is_training,
                                    is_dropout=self.is_dropout) 

class CtrRankModel(regressModel):
      def _build_(self):
            # input coding placeholder
            self.title_source = tf.placeholder(shape=(None, ctrRankModelParams.title_maxlen),dtype=tf.int64)
            self.recall_tag = tf.placeholder(shape=(None,),dtype=tf.int64)
            self.target = tf.placeholder(shape=(None, ),dtype=tf.float32)
            #target to token embedding
            tag_embed = tag_embedding(
                                    inputs=tf.reshape(self.recall_tag,[-1,1]),
                                    vocab_size=ctrRankModelParams.tag_size,
                                    is_training=self.is_training,
                                    scope='recalltag')
            #embedding
            title_embed = semantic_position_embedding(
                                    inputs=self.title_source,
                                    vocab_size=ctrRankModelParams.source_vocab_size,
                                    is_training=self.is_training,
                                    reuse=None,
                                    scope='chinese')
            #title encoding
            title_encoding = sentence_encoder(
                                    inputs=title_embed,
                                    query=tag_embed,                                    
                                    multi_cnn_params=ctrRankModelParams.title_cnn_params,#kernel,stride,layer
                                    scope='title',
                                    is_training=self.is_training,
                                    is_dropout=self.is_dropout,
                                    reuse=None) #N,s,FN

            full_layer = tf.concat([title_encoding,tag_embed],1)            
            self.logits = final_mlp_encoder(
                                    inputs=full_layer,
                                    output_dim=1,
                                    is_training=self.is_training,
                                    is_dropout=self.is_dropout)


class keywordModel(multiClsModel):
      def _build_(self):
            # input coding placeholder
            self.title_source = tf.placeholder(shape=(None, keywordModelParams.title_maxlen),dtype=tf.int64)
            self.target = tf.placeholder(shape=(None,),dtype=tf.int64)
            #target to token embedding
            if keywordModelParams.label_context:
                  context_embed = tag_embedding(
                                          inputs=tf.reshape(self.target,[-1,1]),
                                          vocab_size=keywordModelParams.target_label_num,
                                          is_training=self.is_training,
                                          scope='context_embed')
            else:
                  context_embed = tf.tile(tag_embedding(
                                          inputs=tf.constant([[0]],dtype=tf.int64),
                                          vocab_size=1,
                                          is_training=self.is_training,
                                          scope='context_embed'),[tf.shape(self.target)[0],1,1])
            #embedding
            title_embed = semantic_position_embedding(
                                    inputs=self.title_source,
                                    vocab_size=keywordModelParams.source_vocab_size,
                                    is_training=self.is_training,
                                    position_embed=False,
                                    reuse=None,
                                    scope='chinese')

            keyword_embed,self.weights,self.weights_score = keyword_encoder(
                                    inputs=title_embed,
                                    query=context_embed,
                                    scope='title',
                                    is_training=self.is_training,
                                    is_dropout=self.is_dropout,
                                    reuse=None)

            self.logits = final_mlp_encoder(
                                    inputs=keyword_embed,
                                    output_dim=keywordModelParams.target_label_num,
                                    is_training=self.is_training,
                                    is_dropout=self.is_dropout)

class LDAModel(regressModel):
      def _build_(self):
            # input coding placeholder
            self.user = tf.placeholder(shape=(None,),dtype=tf.int64)
            self.tag = tf.placeholder(shape=(None,),dtype=tf.int64)
            self.target = tf.placeholder(shape=(None,),dtype=tf.float32)
            #target to token embedding
            user_topic_embed = tag_embedding(
                                    inputs=self.user,
                                    vocab_size=LDAModelParams.user_num,
                                    is_training=self.is_training,
                                    scope='user_topic_embed')

            tag_topic_embed = tag_embedding(
                                    inputs=self.tag,
                                    vocab_size=LDAModelParams.tag_num,
                                    is_training=self.is_training,
                                    scope='tag_topic_embed')
            user_topic_dist = tf.nn.softmax(user_topic_embed)
            tag_topic_dist = tf.nn.softmax(tag_topic_embed)
            
            self.logits =  tf.reduce_sum(tf.multiply(user_topic_dist,tag_topic_dist),1)

      def infer(self):
            self.tag = tf.placeholder(shape=(None,),dtype=tf.int64)
            tag_topic_embed = tag_embedding(
                                    inputs=self.tag,
                                    vocab_size=LDAModelParams.tag_num,
                                    is_training=self.is_training,
                                    reuse=True,
                                    scope='tag_topic_embed')
            tag_topic_dist = tf.nn.softmax(tag_topic_embed)

            self.concentration = tf.reduce_max(tag_topic_dist,1)


