from __future__ import print_function
import tensorflow as tf
 
from real2real.modules.text_encoder import text_conv_encoder,text_atten_encoder
from real2real.app.params import ctrRankModelParams,newsClsModelParams
from real2real.models.base_model import regressModel,multiClsModel
from real2real.modules.full_connector import multi_layer_perceptron
from real2real.utils.shape_ops import split_long_text
from pydoc import locate

class ConvRank(regressModel):
            def _build_(self):
                        # input coding placeholder
                        self.source = tf.placeholder(shape=(None, ctrRankModelParams.source_maxlen),dtype=tf.int64)
                        self.tag = tf.placeholder(shape=(None, ),dtype=tf.int64)
                        self.target = tf.placeholder(shape=(None, ),dtype=tf.float32)

                        encoding = text_conv_encoder(
                                                       inputs=self.source,
                                                       vocab_size=ctrRankModelParams.source_vocab_size,
                                                       num_units=ctrRankModelParams.embedding_dim,
                                                       kernel_size=5,
                                                       conv_layer_num=3,
                                                       stride_step=2,
                                                       zero_pad=ctrRankModelParams.zero_pad,
                                                       scale=ctrRankModelParams.scale,
                                                       maxlen=ctrRankModelParams.source_maxlen,
                                                       scope='title',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=None)
                        tag_embed = embedding(
                                          inputs=self.tag,
                                          vocab_size=ctrRankModelParams.tag_size,
                                          num_units=ctrRankModelParams.embedding_dim,
                                          zero_pad=False,
                                          scale=ctrRankModelParams.scale,
                                          scope="tag_embed")
                        #forward feed connect
                        full_layer = tf.concat([tag_embed,encoding],1)
                        self.logits = multi_layer_perceptron(
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
                        
			      title_encoding = text_conv_encoder(
                                                       inputs=self.content_source,
                                                       vocab_size=newsClsModelParams.source_vocab_size,
                                                       num_units=newsClsModelParams.embedding_dim,
                                                       kernel_size=5,
                                                       conv_layer_num=3,
                                                       stride_step=2,
                                                       zero_pad=newsClsModelParams.zero_pad,
                                                       scale=newsClsModelParams.scale,
                                                       maxlen=newsClsModelParams.title_maxlen,
                                                       scope='sentence',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=None) #N,FN

			      split_content,sentence_num = split_long_text(self.content_source,newsClsModelParams.title_maxlen)

                        content_encoding = text_conv_encoder(
                                                       inputs=split_content,
                                                       vocab_size=newsClsModelParams.source_vocab_size,
                                                       num_units=newsClsModelParams.embedding_dim,
                                                       kernel_size=5,
                                                       conv_layer_num=3,
                                                       stride_step=2,
                                                       zero_pad=newsClsModelParams.zero_pad,
                                                       scale=newsClsModelParams.scale,
                                                       maxlen=newsClsModelParams.title_maxlen,
                                                       scope='sentence',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=True)   #N*ST,FN

                        content_encoding = stack_short_encode(content_encoding,sentence_num)#N,ST,FN

                        content_encoding = text_conv_encoder(
                                                       inputs=split_content,
                                                       vocab_size=newsClsModelParams.source_vocab_size,
                                                       num_units=newsClsModelParams.embedding_dim,
                                                       kernel_size=3,
                                                       conv_layer_num=1,
                                                       stride_step=1,
                                                       zero_pad=newsClsModelParams.zero_pad,
                                                       scale=newsClsModelParams.scale,
                                                       maxlen=newsClsModelParams.content_maxlen,
                                                       scope='doc',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=None)   #N,FN
                        #full_layer = tf.concat([title_out,content_out],1)
                        full_layer = content_encoding
                        self.logits = multi_layer_perceptron(
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
                        #target embedding
                        target_embed = tf.get_variable('target_embed',
                                                       dtype=tf.float32,
                                                       shape=[newsClsModelParams.target_vocab_size, newsClsModelParams.embedding_dim],
                                                       initializer=tf.contrib.layers.xavier_initializer(),
                                                       trainable=self.is_training)

                        target_embed = tf.tile(tf.expand_dims(target_embed,0),[tf.shape(self.target)[0],1,1]) #N,m,WD
                        #conv and anttention
                        title_encoding = text_atten_encoder(
                                                       inputs=self.title_source,
                                                       query=target_embed,
                                                       vocab_size=newsClsModelParams.source_vocab_size,
                                                       num_units=newsClsModelParams.embedding_dim,
                                                       kernel_size=3,
                                                       conv_layer_num=1,
                                                       stride_step=1,
                                                       zero_pad=newsClsModelParams.zero_pad,
                                                       scale=newsClsModelParams.scale,
                                                       maxlen=newsClsModelParams.title_maxlen,
                                                       scope='sentence',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=None) #N,m,FN

                        split_content,sentence_num = split_long_text(self.content_source,newsClsModelParams.title_maxlen)

                        content_encoding = text_atten_encoder(
                                                       inputs=split_content,
                                                       query=target_embed,
                                                       vocab_size=newsClsModelParams.source_vocab_size,
                                                       num_units=newsClsModelParams.embedding_dim,
                                                       kernel_size=3,
                                                       conv_layer_num=1,
                                                       stride_step=1,
                                                       zero_pad=newsClsModelParams.zero_pad,
                                                       scale=newsClsModelParams.scale,
                                                       maxlen=newsClsModelParams.content_maxlen,
                                                       scope='sentence',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=True)   #N*ST,m,FN
                        content_encoding = stack_short_encode(content_encoding,sentence_num)#N,ST,m,FN
                        content_encoding = tf.reshape(tf.transpose(content_encoding,[0,2,1,3]),[-1,sentence_num,newsClsModelParams.embedding_dim])#N*m,ST,FN

                        content_encoding = text_atten_encoder(
                                                       inputs=split_content,
                                                       query=target_embed,
                                                       vocab_size=newsClsModelParams.source_vocab_size,
                                                       num_units=newsClsModelParams.embedding_dim,
                                                       kernel_size=1,
                                                       conv_layer_num=1,
                                                       stride_step=1,
                                                       zero_pad=newsClsModelParams.zero_pad,
                                                       scale=newsClsModelParams.scale,
                                                       maxlen=newsClsModelParams.content_maxlen,
                                                       scope='doc',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=None)   ##N*m,m,FN
                        #full connect
                        self.logits = tf.squeeze(tf.layers.dense(content_encoding,1, activation=locate(newsClsModelParams.activation_fn)))

