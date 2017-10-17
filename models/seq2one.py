from __future__ import print_function
import tensorflow as tf
from real2real.layers.conv_layers import multiLayer_conv_strip,conv1d_to_full_layer
from real2real.app.params import convRankParams 
from real2real.models.base_model import regressModel
from real2real.layers.common_layers import semantic_position_embedding,embedding,mlp_layer
from pydoc import locate

class convRank(regressModel):
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
                                                            scope='encoder')

                        conv_2_out = multiLayer_conv_strip(
                                                      inputs=source_embed,
                                                      kernel_size=2,
                                                      conv_layer_num=5,
                                                      scope_name='cnn5',
                                                      is_training=self.is_training,
                                                      is_dropout=self.is_dropout)
                        conv_3_out = multiLayer_conv_strip(
                                                      inputs=source_embed,
                                                      kernel_size=3,
                                                      conv_layer_num=4,
                                                      scope_name='cnn5',
                                                      is_training=self.is_training,
                                                      is_dropout=self.is_dropout)

                        conv_5_out = multiLayer_conv_strip(
                                                      inputs=source_embed,
                                                      kernel_size=5,
                                                      conv_layer_num=3,
                                                      scope_name='cnn5',
                                                      is_training=self.is_training,
                                                      is_dropout=self.is_dropout)
                         
                        conv_2_out = conv1d_to_full_layer(
                                          inputs=conv_2_out,
                                          scope_name="conv2full_2",
                                          is_training=self.is_training)
                        conv_3_out = conv1d_to_full_layer(
                                          inputs=conv_3_out,
                                          scope_name="conv2full_3",
                                          is_training=self.is_training)
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
                        #full_layer = tf.concat([tagEmbed,conv_2_out],1)
                        #full_layer = tf.concat([tagEmbed,conv_2_out,conv_3_out],1)
                        #full_layer = tf.concat([tagEmbed,conv_2_out,conv_3_out,conv_5_out],1)

                        self.logits = mlp_layer(
                                                inputs=full_layer,
                                                output_dim=1,
                                                mlp_layers=convRankParams.mlp_layers,
                                                hidden_units=convRankParams.hidden_units,
						            activation_fn=locate(convRankParams.activation_fn),
                                                is_training=self.is_training,
                                                is_dropout=self.is_dropout)

