from __future__ import print_function
import tensorflow as tf
import sys
import time
import argparse
sys.path.insert(0,'..')
from models.seq2one import convRank
from layers.utils import *
from app.params import baseModelParams,globalParams 

train_cache,test_cache = CreateBatchs()

def training():
        gpu_options = tf.GPUOptions(allow_growth = True)
        model = convRank(is_training=True)
        startTime = time.time()
        with tf.Session(graph = model.graph,config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True, log_device_placement = False)) as sess:
                sess.run(model.init_op_)
                #list all trainable variables the graph hold 
                layout_trainable_variables()
                for epoch in range(baseModelParams.num_epochs):
 
                        for text_code_batch,tag_code_batch,ctr_batch in train_cache:
                                sess.run(model.train_op_,feed_dict={
                                                                model.source:text_code_batch ,
                                                                model.tag:tag_code_batch ,
                                                                model.target:ctr_batch})
 
def evaluation():
        pass

def inference():
        pass
                 

if __name__ == "__main__":

        if baseModelParams.model_mode == 'train':
                training()
        elif baseModelParams.model_mode == 'eval':
                evaluation()
        elif baseModelParams.model_mode == 'infer':
                inference()
        else:
                raise ValueError("model mode must be one of: 'train', 'eval','infer'.")
