from __future__ import print_function
import tensorflow as tf
import sys
import time
import argparse
import os
sys.path.insert(0,"..")
sys.path.append(os.getcwd())

from real2real.models.seq2seq import transformer,simpleAttentionCNN
from real2real.app.params import baseModelParams
from real2real.preprocess.seq2seq_feeds_process import LoadTrainFeeds
from real2real.utils.info_layout import *
from real2real.utils.metrics import regression_model_eval
from pydoc import locate
def training():
        gpu_options = tf.GPUOptions(allow_growth = True)
        #model = transformer(is_training=True)
        model = simpleAttentionCNN(is_training=True)
        cache = LoadTrainFeeds()
        startTime = time.time()
        with tf.Session(graph = model.graph,config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True, log_device_placement = False)) as sess:
                try:
                        model.global_saver.restore(sess,FLAGS.restore_path+"/global_model")
                except:			
                        sess.run(model.init_op)
                #list all trainable variables the graph hold 
                layout_trainable_variables()
                for epoch in range(baseModelParams.num_epochs):
                        iters=0
                        for source,target in cache['training']:
                                iters+=1
                                _,gs=sess.run([model.train_op,model.global_step],feed_dict={
                                                                model.source:source,
                                                                model.target:target,
                                                                model.is_dropout:True})
                                if iters%100==0:
                                        train_acc = sess.run(model.acc,feed_dict={
                                                                model.source:source,                                            
                    						                    model.target:target,
								                                model.is_dropout:False})

                                        train_acc_drop = sess.run(model.acc,feed_dict={
                                                                model.source:source,
                                                                model.target:target,
                                                                model.is_dropout:True})
					                    print('Iteration:%s\ttrain acc:%s\tdrop train acc:%s'%(gs,train_acc,train_acc_drop))
			         #source,target = cache['valid']
                        #test_acc = sess.run(model.acc,feed_dict={
                        #                                        model.source:source,                                                     
                        #                                        model.target:target})

                        #print('Iteration:%s\ttrain acc:%s\ttest acc:%s'%(gs,train_acc,test_acc))
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
