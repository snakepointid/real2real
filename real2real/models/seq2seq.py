from __future__ import print_function
import tensorflow as tf
from real2real.layers.attention_layers import self_attention,enc_dec_attention,conv_attention_conv
from real2real.models.base_model import multiClsModel
from real2real.app.params import transformerParams  
from real2real.layers.common_layers import semantic_position_embedding

class transformer(multiClsModel):
            def _build_(self):
                        # input coding placeholder
                        self.source = tf.placeholder(tf.int32, shape=(None, transformerParams.source_maxlen))
                        self.target = tf.placeholder(tf.int32, shape=(None, transformerParams.target_maxlen))
                  # define decoder inputs
                        decoder_inputs = tf.concat((tf.ones_like(self.target[:, :1])*2, self.target[:, :-1]), -1) # 2:<S>

                        # source embedding
                        encoding = semantic_position_embedding(
                                                            inputs=self.source,
                                                            vocab_size=transformerParams.source_vocab_size,
                                                            num_units=transformerParams.hidden_units,
                                                            maxlen=transformerParams.source_maxlen,
                                                            scope='encoder')
                        # target embedding
                        decoding = semantic_position_embedding(
                                                            inputs=decoder_inputs,
                                                            vocab_size=transformerParams.target_vocab_size,
                                                            num_units=transformerParams.hidden_units,
                                                            maxlen=transformerParams.target_maxlen,
                                                            scope='decoder')
                        # source multi_attention
                        self.encoding = self_attention(
                                                      encoding=encoding,
                                                      is_training=self.is_training,
                                                      is_dropout=self.is_dropout)
                        # target multi_attention
                        self.decoding = enc_dec_attention(
                                                      decoding=decoding,
                                                      encoding=self.encoding,
                                                      is_training=self.is_training,
                                                      is_dropout=self.is_dropout)
                        # Final linear projection
                        self.logits = tf.layers.dense(
                                                      inputs=self.decoding, 
                                                      units=transformerParams.target_vocab_size,
                                                      name="full_conn")
class simpleAttentionCNN(multiClsModel):
            def _build_(self):
                        # input coding placeholder
                        self.source = tf.placeholder(tf.int32, shape=(None, transformerParams.source_maxlen))
                        self.target = tf.placeholder(tf.int32, shape=(None, transformerParams.target_maxlen))
                        # source embedding
                        encoding = semantic_position_embedding(
                                                            inputs=self.source,
                                                            vocab_size=transformerParams.source_vocab_size,
                                                            num_units=transformerParams.hidden_units,
                                                            maxlen=transformerParams.source_maxlen,
                                                            scope='encoder')
                         
                        #simple attention cnn
                        self.decoding=conv_attention_conv(
                                          inputs=encoding,
                                          query_length=transformerParams.target_maxlen,
                                          scope_name="simpAttenCnn",
                                          is_training=self.is_training,
                                          is_dropout=self.is_dropout)
                        # Final linear projection
                        self.logits = tf.layers.dense(
                                                      inputs=self.decoding, 
                                                      units=transformerParams.target_vocab_size,
                                                      name="full_conn")
                        
