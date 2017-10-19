from __future__ import print_function
import tensorflow as tf
from real2real.layers.conv_layers import multiLayer_conv_strip,conv1d_to_full_layer
from real2real.app.params import convRankParams,convClsParams
from real2real.models.base_model import regressModel,multiClsModel
from real2real.layers.common_layers import semantic_position_embedding,embedding,mlp_layer
from pydoc import locate

class ConvRank(regressModel):
            def _build_(self):
                        # input coding placeholder
                        self.source = tf.placeholder(shape=(None, convRankParams.source_maxlen),dtype=tf.int64)
                        self.tag = tf.placeholder(shape=(None, ),dtype=tf.int64)
                        self.target = tf.placeholder(shape=(None, ),dtype=tf.float32)
                        source_embed = semantic_position_embedding(
                                                            inputs=self.source,
                                                            vocab_size=convRankParams.source_vocab_size,
                                                            num_units=convRankParams.embedding_dim,
                                                            maxlen=convRankParams.source_maxlen,
                                                            scope='encoder',
							    reuse=None)

                        conv_5_out = multiLayer_conv_strip(
                                                      inputs=source_embed,
                                                      kernel_size=5,
                                                      conv_layer_num=3,
                                                      stride_step=2,
                                                      scope_name='cnn5',
                                                      is_training=self.is_training,
                                                      is_dropout=self.is_dropout)
                         
                        conv_5_out = conv1d_to_full_layer(
                                          inputs=conv_5_out,
                                          scope_name="conv2full_5",
                                          is_training=self.is_training)

                        tagEmbed = embedding(
                                          inputs=self.tag,
                                          vocab_size=convRankParams.tag_size,
                                          num_units=convRankParams.embedding_dim,
                                          zero_pad=False,
                                          scale=True,
                                          scope="tagEmbed")

                        full_layer = tf.concat([tagEmbed,conv_5_out],1)
                        self.logits = mlp_layer(
                                                inputs=full_layer,
                                                output_dim=1,
                                                mlp_layers=convRankParams.mlp_layers,
                                                hidden_units=convRankParams.hidden_units,
                                                activation_fn=locate(convRankParams.activation_fn),
                                                is_training=self.is_training,
                                                is_dropout=self.is_dropout)

class ConvCls(multiClsModel):
            def _build_(self):
                        # input coding placeholder
                        self.title_source = tf.placeholder(shape=(None, convClsParams.title_maxlen),dtype=tf.int64)
                        self.content_source = tf.placeholder(shape=(None, convClsParams.content_maxlen),dtype=tf.int64)
                        self.target = tf.placeholder(shape=(None, ),dtype=tf.int32)

                        title_embed = semantic_position_embedding(
                                                            inputs=self.title_source,
                                                            vocab_size=convClsParams.source_vocab_size,
                                                            num_units=convClsParams.embedding_dim,
                                                            maxlen=convClsParams.title_maxlen,
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
                                                            vocab_size=convClsParams.source_vocab_size,
                                                            num_units=convClsParams.embedding_dim,
                                                            maxlen=convClsParams.content_maxlen,
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

                         

                        full_layer = tf.concat([title_out,content_out],1)
                        #full_layer = content_out
			self.logits = mlp_layer(
                                                inputs=full_layer,
                                                output_dim=convClsParams.target_vocab_size,
                                                mlp_layers=convClsParams.mlp_layers,
                                                hidden_units=convClsParams.hidden_units,
                                                activation_fn=locate(convClsParams.activation_fn),
                                                is_training=self.is_training,
                                                is_dropout=self.is_dropout)

