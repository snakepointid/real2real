from __future__ import print_function
import tensorflow as tf

from real2real.models.base_model import regressModel,multiClsModel

from real2real.modules.text_encoder import sentence_encoder
from real2real.modules.full_connector import final_mlp_encoder
from real2real.modules.entity_encoder import *

from real2real.layers.rnn_layers import uni_lstm
from real2real.utils.shape_ops import *

from real2real.app.params import nmtModelParams,languageModelParams
   
class NmtModel (multiClsModel):
      def _build_(self):
            # input coding placeholder
            self.source_code = tf.placeholder(shape=(None, nmtModelParams.source_maxlen),dtype=tf.int64)
            self.target_code = tf.placeholder(shape=(None, nmtModelParams.target_maxlen),dtype=tf.int64)
                        
            #embedding
            source_embed = semantic_position_embedding(
                                                       inputs=self.source_code,
                                                       vocab_size=nmtModelParams.source_vocab_size,
                                                       is_training=self.is_training,
                                                       reuse=None,
                                                       scope=nmtModelParams.language)
                        
            #title encoding
            source_encoding = sentence_encoder(
                                                       inputs=source_embed,
                                                       query=None,
                                                       layers='CP',                                                       
                                                       multi_cnn_params=nmtModelParams.source_cnn_params,#kernel,stride,layer
                                                       scope='source',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=None) #N,FN
            #corse predict
            self.logits = final_mlp_encoder(
                                                       inputs=source_encoding,
                                                       output_dim=nmtModelParams.target_label_num,
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout) #N,tar

            self.target = tf.one_hot(indices=self.target_code,depth=nmtModelParams.target_label_num)#N,Len,De
            self.target = tf.reduce_sum(self.target,1)
            self.target = tf.to_float(tf.not_equal(self.target, 0))
            self.target = tf.reshape(self.target,[-1,1])

class LanguageModel(multiClsModel):
      def _build_(self):
            # input coding placeholder
            self.source_code = tf.placeholder(shape=(None, nmtModelParams.source_maxlen),dtype=tf.int64)

            
            self.input = self.source_code[:,:-1]
            self.target = self.source_code[:,1:]
                        
            #embedding
            source_embed = semantic_position_embedding(
                                                       inputs=self.input,
                                                       vocab_size=languageModelParams.source_vocab_size,
                                                       is_training=self.is_training,
                                                       reuse=None,
                                                       scope=languageModelParams.language)
                        
            rnn_outptus = uni_lstm(
                              inputs=source_embed,
                              reuse=None,
                              is_training=self.is_training)#N,SL,D
            #corse predict
            self.logits = tf.layers.dense(rnn_outptus, languageModelParams.source_vocab_size, activation=tf.nn.relu)#N,SL,out
            self.logits = tf.reshape(self.logits,[-1,languageModelParams.source_vocab_size])
	    self.target = tf.reshape(self.target,[-1,1])
     
         
