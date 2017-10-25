from __future__ import print_function
import tensorflow as tf

from real2real.models.base_model import regressModel,multiClsModel

from real2real.modules.text_encoder import sentence_encoder
from real2real.modules.full_connector import final_mlp_encoder
from real2real.modules.entity_encoder import *

from real2real.utils.shape_ops import *

from real2real.app.params import nmtModelParams
   
class NmtModel (multiClsModel):
            def _build_(self):
                        # input coding placeholder
                        self.source_source = tf.placeholder(shape=(None, nmtModelParams.source_maxlen),dtype=tf.int64)
                        self.target_source = tf.placeholder(shape=(None, nmtModelParams.target_maxlen),dtype=tf.int64)
                        #embedding
                        source_embed = semantic_position_embedding(
                                                       inputs=self.source_source,
                                                       vocab_size=nmtModelParams.source_vocab_size,
                                                       is_training=self.is_training,
                                                       reuse=None,
                                                       scope='chinese')
                        #embedding
                        target_embed = semantic_position_embedding(
                                                       inputs=self.target_source,
                                                       vocab_size=nmtModelParams.source_vocab_size,
                                                       is_training=self.is_training,
                                                       reuse=None,
                                                       scope='english')
                        #title encoding
                        source_encoding = sentence_encoder(
                                                       inputs=source_embed,
                                                       query=token_context,                                                       
                                                       multi_cnn_params=nmtModelParams.source_cnn_params,#kernel,stride,layer
                                                       scope='title',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=None) #N,FN
                         
                        target_encoding = sentence_encoder(
                                                       inputs=target_embed,
                                                       query=token_context, 
                                                       multi_cnn_params=nmtModelParams.target_cnn_params,#kernel,stride,layer
                                                       scope='content',
                                                       is_training=self.is_training,
                                                       is_dropout=self.is_dropout,
                                                       reuse=None)#N,FN
                        
 
                        self.logits = final_mlp_encoder(
                                             inputs=full_layer,
                                             output_dim=nmtModelParams.target_label_num,
                                             is_training=self.is_training,
                                             is_dropout=self.is_dropout) 

