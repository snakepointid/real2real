from __future__ import print_function
import tensorflow as tf
from real2real.models.base_model import *
from real2real.app.params import tokenEmbedModelParams

from real2real.modules.entity_encoder import *
from real2real.modules.full_connector import final_mlp_encoder
from real2real.modules.text_encoder import *

from real2real.utils.tensor_ops import *
class TokenEmbed(binomialModel):
            def _build_(self):
                        # input coding placeholder
                        self.pair  = tf.placeholder(tf.int32, shape=(None,2))
                        target_token ,context_token= tf.split(self.pair,[1,1],1)
                        self.target = tf.placeholder(tf.float32 , shape=(None,))
                        #embedding
                        with tf.variable_scope(tokenEmbedModelParams.language,reuse=None):
                                    target_embed = tag_embedding(
                                                            inputs=target_token,
                                                            vocab_size=tokenEmbedModelParams.source_vocab_size,
                                                            is_training=self.is_training,
                                                            scope='token',
                                                            reuse=None)
                                    context_embed = tag_embedding(
                                                            inputs=context_token,
                                                            vocab_size=tokenEmbedModelParams.source_vocab_size,
                                                            is_training=self.is_training,
                                                            scope='weight',
                                                            reuse=None)

                        self.logits = tf.reduce_sum(tf.multiply(target_embed,context_embed),[1,2])

            def infer(self):
                        with tf.variable_scope(tokenEmbedModelParams.language,reuse=True):
                                    token_embed = tf.get_variable("token/lookup_table")                                     
                                    self.query = tf.placeholder(tf.int64, shape=(None,))
                                    qeury_embed = tag_embedding(
                                                            inputs=self.query,
                                                            vocab_size=tokenEmbedModelParams.source_vocab_size,
                                                            is_training=self.is_training,
                                                            scope='token',
                                                            reuse=True)

                        self.most_k_similar = compute_similariry(qeury_embed,token_embed,tokenEmbedModelParams.topk,'dot')         

             
 

 



