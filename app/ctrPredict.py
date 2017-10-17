from __future__ import print_function
import tensorflow as tf
import sys
import time
import argparse
import os
sys.path.insert(0,"..")
sys.path.append(os.getcwd())
print(sys.path)
from real2real.models.seq2one import *
from real2real.app.params import baseModelParams
from real2real.preprocess.seq2one_feeds_process import LoadTrainFeeds
from real2real.utils.info_layout import *
from real2real.utils.metrics import regression_model_eval

def training():
        gpu_options = tf.GPUOptions(allow_growth = True)
        model = convRank(is_training=True)
        startTime = time.time()
        with tf.Session(graph = model.graph,config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True, log_device_placement = False)) as sess:
                try:
                        model.global_saver.restore(sess,FLAGS.restore_path+"/global_model")
                except:			
                        sess.run(model.init_op)
			print("initial the graph")
		        #list all trainable variables the graph hold 
                layout_trainable_variables()
                cache = LoadTrainFeeds()
                #compute the initial pearson coef
		text_code_batch,tag_code_batch,ctr_batch = cache['testa']   
                probs = sess.run(model.logits,feed_dict={
                                                        model.source:text_code_batch,                                            
                                                        model.tag:tag_code_batch,
                                                        model.is_dropout:False})
                old_pc = regression_model_eval(ctr_batch,probs,'testa',False)

                for epoch in range(baseModelParams.num_epochs):
                        for text_code_batch,tag_code_batch,ctr_batch in cache['training']:
                                _,gs=sess.run([model.train_op,model.global_step],feed_dict={
                                                                model.source:text_code_batch ,
                                                                model.tag:tag_code_batch ,
                                                                model.target:ctr_batch,
								model.is_dropout:True})
	 
                        text_code_batch,tag_code_batch,ctr_batch = cache['train']	
                        probs = sess.run(model.logits,feed_dict={
                                                model.source:text_code_batch,                                            
                            			model.tag:tag_code_batch,
        			                model.is_dropout:False})

                        _ = regression_model_eval(ctr_batch,probs,'train')
                        text_code_batch,tag_code_batch,ctr_batch = cache['valid']   
                        probs = sess.run(model.logits,feed_dict={
                                            			model.source:text_code_batch,                                            
                                                                model.tag:tag_code_batch,
                                                                model.is_dropout:False})

                        _ = regression_model_eval(ctr_batch,probs,'valid')

                        text_code_batch,tag_code_batch,ctr_batch = cache['testa']   
                        probs = sess.run(model.logits,feed_dict={
                                                                model.source:text_code_batch,                                            
                                                                model.tag:tag_code_batch,
                                                                model.is_dropout:False})

                        new_pc = regression_model_eval(ctr_batch,probs,'testa')
			if new_pc<old_pc:
				break
                   	old_pc = new_pc     
                        endTime = time.time()
                        if endTime-startTime>3600:
                                print ("save the whole model")
                                model.global_saver.save(sess,FLAGS.save_path+"/global_model")
                                startTime = endTime

                model.global_saver.save(sess,FLAGS.save_path+"/global_model")
 
def evaluation():
        pass

def inference():
        gpu_options = tf.GPUOptions(allow_growth = True)
        model = convRank(is_training=False)
        with tf.Session(graph = model.graph,config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True, log_device_placement = False)) as sess:
                model.global_saver.restore(sess,FLAGS.restore_path+"/global_model")
                cache = LoadPredictFeeds()
    
                for raw_batch,source_batch,tag_batch in cache:
                        probs=sess.run(model.logits,feed_dict={
                                                        model.source:text_code_batch ,
                                                        model.tag:tag_code_batch ,
                                                        model.is_dropout:False}) 

                        for idx,raw in enumerate(raw_batch):
                                print("%s\t%s"%(raw,probs[idx]))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
        parser.register("type", "bool", lambda v: v.lower() == "true")
        # Flags for defining the parameter of data path
        parser.add_argument(
                "--save_path",
                type=str,
                default="",
                help="The save path for model"
        )
        parser.add_argument(
                "--restore_path",
                type=str,
                default="",
                help="The save path for model"
        )
        parser.add_argument(
                "--model_mode",
                type=str,
                default="train",
                help="The save path for model"
        )
        FLAGS, unparsed = parser.parse_known_args()

        if FLAGS.model_mode == 'train':
                training()
        elif FLAGS.model_mode == 'eval':
                evaluation()
        elif FLAGS.model_mode == 'infer':
                inference()
        else:
                raise ValueError("model mode must be one of: 'train', 'eval','infer'.")
