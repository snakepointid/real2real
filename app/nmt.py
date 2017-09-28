from __future__ import print_function
import tensorflow as tf
import sys
import time
import argparse
sys.path.insert(0,'..')
from models.seq2seq import transformer
from layers.utils import *
from app.params import baseModelParams as hp 

def training():
        gpu_options = tf.GPUOptions(allow_growth = True)
        g = transformer(is_training=True)
        startTime = time.time()
        with tf.Session(graph = g.graph,config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True, log_device_placement = False)) as sess:
                sess.run(g.init_op_)
                #print "list all variables the graph hold!!!"
                layout_trainable_variables()

def evaluation():
        pass

def inference():
        pass
                 

if __name__ == "__main__":
        if hp.model_mode == 'train':
                training()
        elif hp.model_mode == 'eval':
                evaluation()
        elif hp.model_mode == 'infer':
                inference()
        else:
                raise ValueError("model mode must be one of: 'train', 'eval','infer'.")