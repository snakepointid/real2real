from __future__ import print_function
import tensorflow as tf
from real2real.layers.conv_layers import multiLayer_conv_strip,conv1d_to_full_layer
from real2real.layers.common_layers import semantic_position_embedding,embedding,mlp_layer
from real2real.layers.attention_layers import multi_hot_attention

from real2real.app.params import ctrRankModelParams,newsClsModelParams
from real2real.models.base_model import regressModel,multiClsModel
from pydoc import locate

class ConvRank(regressModel):
            def _build_(self):
                        # input coding placeholder
                        self.source = tf.placeholder(shape=(None, ctrRankModelParams.source_maxlen),dtype=tf.int64)
                        self.tag = tf.placeholder(shape=(None, ),dtype=tf.int64)
                        self.target = tf.placeholder(shape=(None, ),dtype=tf.float32)
                        #embeddings
                        source_embed = semantic_position_embedding(
                                                            inputs=self.source,
                                                            vocab_size=ctrRankModelParams.source_vocab_size,
                                                            num_units=ctrRankModelParams.embedding_dim,
                                                            is_training=self.is_training,
                                                            maxlen=ctrRankModelParams.source_maxlen,
                                                            scope='encoder',
                                                            reuse=None)
                        tag_embed = embedding(
                                          inputs=self.tag,
                                          vocab_size=ctrRankModelParams.tag_size,
                                          num_units=ctrRankModelParams.embedding_dim,
                                          zero_pad=False,
                                          scale=True,
                                          scope="tag_embed")
                        #convolution
                        conv_out = multiLayer_conv_strip(
                                                      inputs=source_embed,
                                                      kernel_size=5,
                                                      conv_layer_num=3,
                                                      stride_step=2,
                                                      scope_name='cnn',
                                                      is_training=self.is_training,
                                                      is_dropout=self.is_dropout)
                         
                        conv_out = conv1d_to_full_layer(
                                                      inputs=conv_out,
                                                      scope_name="conv2full",
                                                      is_training=self.is_training)

                        
                        #forward feed connect
                        full_layer = tf.concat([tag_embed,conv_out],1)
                        self.logits = mlp_layer(
                                                inputs=full_layer,
                                                output_dim=1,
                                                mlp_layers=ctrRankModelParams.mlp_layers,
                                                hidden_units=ctrRankModelParams.hidden_units,
                                                activation_fn=ctrRankModelParams.activation_fn,
                                                is_training=self.is_training,
                                                is_dropout=self.is_dropout)

 
class ConvCls(multiClsModel):
            def _build_(self):
                        # input coding placeholder
                        self.title_source = tf.placeholder(shape=(None, newsClsModelParams.title_maxlen),dtype=tf.int64)
                        self.content_source = tf.placeholder(shape=(None, newsClsModelParams.content_maxlen),dtype=tf.int64)
                        self.target = tf.placeholder(shape=(None, ),dtype=tf.int32)

                        title_embed = semantic_position_embedding(
                                                            inputs=self.title_source,
                                                            vocab_size=newsClsModelParams.source_vocab_size,
                                                            num_units=newsClsModelParams.embedding_dim,
                                                            is_training=self.is_training,
                                                            maxlen=newsClsModelParams.title_maxlen,
                                                            scope='zh_encode',
                                                            reuse=None)

                        title_out = multiLayer_conv_strip(
                                                      inputs=title_embed,
                                                      kernel_size=5,
                                                      stride_step=2,
                                                      conv_layer_num=3,
                                                      scope_name='cnn_title',
                                                      is_training=self.is_training,
                                                      is_dropout=self.is_dropout)

                        content_embed = semantic_position_embedding(
                                                            inputs=self.content_source,
                                                            vocab_size=newsClsModelParams.source_vocab_size,
                                                            num_units=newsClsModelParams.embedding_dim,
                                                            is_training=self.is_training,
                                                            maxlen=newsClsModelParams.content_maxlen,
                                                            scope='zh_encode',
                                                            reuse=True)

                        content_out = multiLayer_conv_strip(
                                                      inputs=content_embed,
                                                      kernel_size=5,
                                                      stride_step=3,
                                                      conv_layer_num=6,
                                                      scope_name='cnn_content',
                                                      is_training=self.is_training,
                                                      is_dropout=self.is_dropout)
                         
                        title_out = conv1d_to_full_layer(
                                          inputs=title_out,
                                          scope_name="title2full",
                                          is_training=self.is_training)

                        content_out = conv1d_to_full_layer(
                                          inputs=content_out,
                                          scope_name="content2full",
                                          is_training=self.is_training)            

                        #full_layer = tf.concat([title_out,content_out],1)
                        full_layer = content_out
                        self.logits = mlp_layer(
                                          inputs=full_layer,
                                          output_dim=newsClsModelParams.target_vocab_size,
                                          mlp_layers=newsClsModelParams.mlp_layers,
                                          hidden_units=newsClsModelParams.hidden_units,
                                          activation_fn=newsClsModelParams.activation_fn,
                                          is_training=self.is_training,
                                          is_dropout=self.is_dropout)
class AttenCls(multiClsModel):
            def _build_(self):
                        # input coding placeholder
                        self.title_source = tf.placeholder(shape=(None, newsClsModelParams.title_maxlen),dtype=tf.int64)
                        self.content_source = tf.placeholder(shape=(None, newsClsModelParams.content_maxlen),dtype=tf.int64)
                        self.target = tf.placeholder(shape=(None, ),dtype=tf.int32)
                        #embedding
                        title_embed = semantic_position_embedding(
                                                            inputs=self.title_source,
                                                            vocab_size=newsClsModelParams.source_vocab_size,
                                                            num_units=newsClsModelParams.embedding_dim,
                                                            is_training=self.is_training,
                                                            maxlen=newsClsModelParams.title_maxlen,
                                                            scope='zh_encode',
                                                            reuse=None)

                        content_embed = semantic_position_embedding(
                                                            inputs=self.content_source,
                                                            vocab_size=newsClsModelParams.source_vocab_size,
                                                            num_units=newsClsModelParams.embedding_dim,
                                                            is_training=self.is_training,
                                                            maxlen=newsClsModelParams.content_maxlen,
                                                            scope='zh_encode',
                                                            reuse=True)

                        target_embed = tf.get_variable('stack_var',
                                                       dtype=tf.float32,
                                                       shape=[newsClsModelParams.target_vocab_size, newsClsModelParams.embedding_dim],
                                                       initializer=tf.contrib.layers.xavier_initializer(),
                                                       trainable=self.is_training)

                        target_embed = tf.tile(tf.expand_dims(target_embed,0),[tf.shape(self.target)[0],1,1]) #N,m,WD
                        #conv and anttention
                        title_out = multiLayer_conv_strip(
                                                      inputs=title_embed,
                                                      kernel_size=3,
                                                      stride_step=1,
                                                      conv_layer_num=1,
                                                      scope_name='cnn_title',
                                                      is_training=self.is_training,
                                                      is_dropout=self.is_dropout)

                        content_out = multiLayer_conv_strip(
                                                      inputs=content_embed,
                                                      kernel_size=3,
                                                      stride_step=1,
                                                      conv_layer_num=1,
                                                      scope_name='cnn_content',
                                                      is_training=self.is_training,
                                                      is_dropout=self.is_dropout)
                         
                        title_atten_out = multi_hot_attention(
                                                      inputs=title_out,
                                                      query=target_embed,
                                                      scope_name="multi_hot_atten",
                                                      is_training=self.is_training) #N,m,WD

                        content_atten_out = multi_hot_attention(
                                                      inputs=content_out,
                                                      query=target_embed,
                                                      scope_name="multi_hot_atten",
                                                      is_training=self.is_training) #N,m,WD

                        self.logits = tf.squeeze(tf.layers.dense(content_atten_out,1, activation=locate(newsClsModelParams.activation_fn)))

