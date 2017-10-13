from __future__ import print_function
import tensorflow as tf
from real2real.models.base_model import multiClsModel
from real2real.app.params import pairEmbedModelParms

class pairEmbed(multiClsModel):
            def _build_(self):
                        # input coding placeholder
                        self.pair = tf.placeholder(tf.int32, shape=(None,2))
                        self.target = tf.placeholder(tf.float32 , shape=(None, 1))
                        #embed
                        pair_embed = embedding(
                                                inputs=self.pair,
                                                vocab_size=pairEmbedModelParms.vocab_size,
                                                num_units=pairEmbedModelParms.embedding_dim,
                                                zero_pad=False,
                                                scale=True,
                                                scope="pair")
                        #reshape
                        full_layer = tf.reshape(pair_embed,[-1,2*pairEmbedModelParms.embedding_dim])
                        self.logits = mlp_layer(
                                                inputs=full_layer,
                                                output_dim=1,
                                                mlp_layers=pairEmbedModelParms.mlp_layers,
                                                hidden_units=pairEmbedModelParms.hidden_units,
                                                activation_fn=locate(pairEmbedModelParms.activation_fn),
                                                is_training=self.is_training,
                                                is_dropout=self.is_dropout)
                         
 
                        
