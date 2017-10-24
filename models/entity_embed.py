from __future__ import print_function
import tensorflow as tf
from real2real.models.base_model import multiClsModel
from real2real.app.params import tokenEmbedModelParams

class TokenEmbed(multiClsModel):
            def _build_(self):
                        # input coding placeholder
                        self.pair=tf.placeholder(tf.int32, shape=(None,2))
                        self.target=tf.placeholder(tf.float32 , shape=(None,))
                        #embedding
                        pair_embed = semantic_position_embedding(
                                                       inputs=self.pair,
                                                       vocab_size=tokenEmbedModelParams.source_vocab_size,
                                                       is_training=self.is_training,
                                                       position_embed=False
                                                       reuse=None,
                                                       scope=tokenEmbedModelParams.language)
                        #reshape
                        full_layer=pair_embed

                        self.logits = final_mlp_encoder(
                                             inputs=full_layer,
                                             output_dim=tokenEmbedModelParams.target_label_num,
                                             is_training=self.is_training,
                                             is_dropout=self.is_dropout) 
