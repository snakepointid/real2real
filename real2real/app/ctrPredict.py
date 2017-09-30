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
from pydoc import locate
def training():
        gpu_options = tf.GPUOptions(allow_growth = True)
        model = convRank(is_training=True)
        #model = attentionRank(is_training=True)
        startTime = time.time()
        with tf.Session(graph = model.graph,config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True, log_device_placement = False)) as sess:
                try:
			model.global_saver.restore(sess,FLAGS.restore_path+"/global_model")
		except:			
			sess.run(model.init_op)
		#list all trainable variables the graph hold 
		layout_trainable_variables()
		cache = LoadTrainFeeds()
                for epoch in range(baseModelParams.num_epochs):
                        for text_code_batch,tag_code_batch,ctr_batch in cache['training']:
                                _,gs=sess.run([model.train_op,model.global_step],feed_dict={
                                                                model.source:text_code_batch ,
                                                                model.tag:tag_code_batch ,
                                                                model.target:ctr_batch,
								model.is_dropout:True})
			is_dropout = True
			text_code_batch,tag_code_batch,ctr_batch = cache['train']	
			probs_ = sess.run(model.logits,feed_dict={
                                                                model.source:text_code_batch,                                            
                    						model.tag:tag_code_batch,
								model.is_dropout:is_dropout})

                        regression_model_eval(ctr_batch,probs_,'train')
			text_code_batch,tag_code_batch,ctr_batch = cache['valid']   
                        probs_ = sess.run(model.logits,feed_dict={
                                                                model.source:text_code_batch,                                            
                                                                model.tag:tag_code_batch,
                                                                model.is_dropout:is_dropout})

                        regression_model_eval(ctr_batch,probs_,'valid')
                        print('Iteration:%s'%gs)
                        endTime = time.time()
                        if endTime-startTime>3600:
                                print ("save the whole model")
                                model.global_saver.save(sess,FLAGS.save_path+"/global_model")
                                startTime = endTime

                model.global_saver.save(sess,FLAGS.save_path+"/global_model")
 
def evaluation():
        pass

def inference():
        pass
                 

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
        FLAGS, unparsed = parser.parse_known_args()

        if baseModelParams.model_mode == 'train':
                training()
        elif baseModelParams.model_mode == 'eval':
                evaluation()
        elif baseModelParams.model_mode == 'infer':
                inference()
        else:
                raise ValueError("model mode must be one of: 'train', 'eval','infer'.")
