from __future__ import print_function
import tensorflow as tf
import sys
import time
import argparse
import os
sys.path.insert(0,"..")
sys.path.append(os.getcwd())
from real2real.models.entity_embed import TokenEmbed
from real2real.app.params import appParams
from real2real.preprocess.news_cls_feeds import *
from real2real.utils.info_layout import *

def training():
        gpu_options = tf.GPUOptions(allow_growth = True)

        model = TokenEmbed(is_training=True)

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

                for epoch in range(appParams.num_epochs):
                        for pair_batch,label_batch in cache['training']:
                                _,gs=sess.run([model.train_op,model.global_step],feed_dict={
                                                                model.pair:pair_batch , 
                                                                model.target:label_batch,
                                                                model.is_dropout:True})
	 
                        pair_batch,label_batch = cache['train']	
                        train_acc,train_loss = sess.run([model.acc,model.mean_loss],feed_dict={
                                                                model.pair:pair_batch,      
                                                                model.target:label_batch,
                                                                model.is_dropout:False})

                        train_num=len(label_batch)
                        pair_batch,label_batch = cache['valid']   
                        valid_acc,valid_loss = sess.run([model.acc,model.mean_loss],feed_dict={
                                                                model.pair:pair_batch,                                                                                                           
                                                                model.target:label_batch,
                                                                model.is_dropout:False})
                        valid_num=len(label_batch)
                        print("Iteration:%s\ttrain num:%s\ttrain acc:%s\ttrain loss:%s\tvalid num:%s\tvalid acc:%s\tvalid loss:%s"%(gs,train_num,train_acc,train_loss,valid_num,valid_acc,valid_loss))  
                        endTime = time.time()
                        if endTime-startTime>3600:
                                print ("save the whole model")
                                model.global_saver.save(sess,FLAGS.save_path+"/global_model")
                                startTime = endTime
                                
                print ("save the whole model")
                model.global_saver.save(sess,FLAGS.save_path+"/global_model")
 
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
        else:
                raise ValueError("model mode must be 'train'.")
