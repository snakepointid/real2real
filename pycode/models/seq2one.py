from __future__ import print_function
import tensorflow as tf
from layers.conv_layers import multiLayer_conv_strip,conv1d_to_full_layer
from app.params import convRankParams 
from models.base_model import regressModel
from layers.common_layers import semantic_position_embedding,embedding,mlp_layer
from layers.attention_layers import attention_conv
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

                        conv_c_out = multiLayer_conv_strip(
                                                      inputs=source_embed,
                                                      is_training=self.is_training,
                                                      is_dropout=self.is_dropout)
			self.encoding = conv1d_to_full_layer(
							inputs=conv_c_out,
							scope_name="conv2full",
							is_training=self.is_training)
                        tagEmbed = embedding(
                                          inputs=self.tag,
                                          vocab_size=convRankParams.tag_size,
                                          num_units=convRankParams.embedding_dim,
                                          zero_pad=False,
                                          scale=True,
                                          scope="tagEmbed")

                        full_layer = tf.concat([tagEmbed,self.encoding],1)
                        self.logits = mlp_layer(
                                                inputs=full_layer,
                                                output_dim=1,
                                                mlp_layers=convRankParams.mlp_layers,
                                                hidden_units=convRankParams.hidden_units,
						activation_fn=locate(convRankParams.activation_fn),
                                                is_training=self.is_training,
                                                is_dropout=self.is_dropout)
                        # saver
                        self.global_saver = tf.train.Saver()

class attentionRank(regressModel):
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

                        # source multi_attention
                        encoding = attention_conv(
                                                 encoding=source_embed,
                                                 is_training=self.is_training,
                                                 is_dropout=self.is_dropout)
			
			self.encoding = conv1d_to_full_layer(
                                                        inputs=encoding,
                                                        scope_name="conv2full",
                                                        is_training=self.is_training)	
			tagEmbed = embedding(
                                          inputs=self.tag,
                                          vocab_size=convRankParams.tag_size,
                                          num_units=convRankParams.embedding_dim,
                                          zero_pad=False,
                                          scale=True,
                                          scope="tagEmbed")

                        full_layer = tf.concat([tagEmbed,self.encoding],1)
                        self.logits = mlp_layer(
                                                inputs=full_layer,
                                                output_dim=1,
                                                mlp_layers=convRankParams.mlp_layers,
                                                hidden_units=convRankParams.hidden_units,
                                                activation_fn=locate(convRankParams.activation_fn),
                                                is_training=self.is_training,
                                                is_dropout=self.is_dropout)
                        # saver
                        self.global_saver = tf.train.Saver()
