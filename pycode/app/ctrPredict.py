from __future__ import print_function
import tensorflow as tf
import sys
import time
import argparse
import os
sys.path.append(os.getcwd())
sys.path.insert(0,'..')
from models.seq2one import convRank
from layers.utils import *
from app.params import baseModelParams
from preprocess.seq2one_feeds_process import LoadTrainFeeds
from utils.info_layout import *
from utils.metrics import ctrEval
def training():
        gpu_options = tf.GPUOptions(allow_growth = True)
        model = convRank(is_training=True)
	reader = LoadTrainFeeds()
        startTime = time.time()
        with tf.Session(graph = model.graph,config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True, log_device_placement = False)) as sess:
                sess.run(model.init_op_)
                #list all trainable variables the graph hold 
                layout_trainable_variables()
                for epoch in range(baseModelParams.num_epochs):
                        for flag,cache in reader:
				text_code_batch,tag_code_batch,ctr_batch = cache
				if flag=='test':
					break
                                _,gs=sess.run([model.train_op,model.global_step],feed_dict={
                                                                model.source:text_code_batch ,
                                                                model.tag:tag_code_batch ,
                                                                model.target:ctr_batch,
								model.is_dropout:True})
			probs_ = sess.run(model.logits,feed_dict={
                                                                model.source:text_code_batch,
                                                                model.tag:tag_code_batch})
                        ctrEval(ctr_batch,probs_,flag)
                        print('Iteration:%s'%gs)
                        endTime = time.time()
                        if endTime-startTime>3600:
                                print ("save the whole model")
                                g.global_saver.save(sess,FLAGS.save_path+"/global_model")
                                startTime = endTime

                g.global_saver.save(sess,FLAGS.save_path+"/global_model")
 
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
