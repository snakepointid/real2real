from __future__ import print_function
import tensorflow as tf
import sys
import time
import argparse
import os
sys.path.insert(0,"..")
sys.path.append(os.getcwd())
from real2real.models.seq2one import LDAModel
from real2real.app.params import baseModelParams
from real2real.preprocess.lda_feeds import *
from real2real.utils.info_layout import *
from real2real.utils.metrics import regression_model_eval

def training():
      gpu_options = tf.GPUOptions(allow_growth = True)
      model = LDAModel(is_training=True)
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
            old_loss = 1
            #compute the initial pearson coef
            for epoch in range(baseModelParams.num_epochs):
                  for user_batch,tag_batch,label_batch in cache['training']:
                        _,gs=sess.run([model.train_op,model.global_step],feed_dict={
                                                            model.user:user_batch ,
                                                            model.tag:tag_batch,
                                                            model.target:label_batch,
                                                            model.is_dropout:True})
	 
                  user_batch,tag_batch,label_batch = cache['train']	
                  train_num = len(tag_batch)
                  train_loss = sess.run(model.mean_loss,feed_dict={
                                                            model.user:user_batch, 
                                                            model.tag:tag_batch,
                                                            model.target:label_batch,                                           
                                                            model.is_dropout:False})

                  user_batch,tag_batch,label_batch = cache['valid']   
                  test_num = len(tag_batch)
                  test_loss = sess.run(model.mean_loss,feed_dict={
                                                            model.user:user_batch,
                                                            model.tag:tag_batch,
                                                            model.target:label_batch,                                            
                                                            model.is_dropout:False})                  
                  print("iteration:%s\ttrain num:%s\ttrain loss:%s\ttest num:%s\ttest loss:%s"%(gs,train_num,train_loss,test_num,test_loss))
                  endTime = time.time()
                  if endTime-startTime>1000:
                        print ("save the whole model")
                        model.global_saver.save(sess,FLAGS.save_path+"/global_model")
                        startTime = endTime
            print ("save the whole model")
            model.global_saver.save(sess,FLAGS.save_path+"/global_model")
 
def inference():
      tag2code = pickle.load(open("/home/hdp-reader-tag/shechanglue/index_eval/tag_eval/res/tag2code.pkl","rb"))
      tag2code['UNK']=0
      code2tag = dict(zip(tag2code.values(), tag2code.keys()))
      
      gpu_options = tf.GPUOptions(allow_growth = True)
      model = LDAModel(is_training=False)
      tags = np.arange(9000)
      with tf.Session(graph = model.graph,config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True, log_device_placement = False)) as sess:
            model.global_saver.restore(sess,FLAGS.restore_path+"/global_model")        
      	    model.infer()
            concentration = sess.run(model.concentration,feed_dict={model.tag:tags})
            for idx,tagcode in enumerate(tags):
                tag = code2tag.get(tagcode,0)
                if tag:
                    print("%s\t%s"%(tag,concentration[idx]))    

if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.register("type", "bool", lambda v: v.lower() == "true")
      # Flags for defining the parameter of data path
      parser.add_argument(
                "--save_path",
                type=str,
                default="",
                help="The save path for model")

      parser.add_argument(
                "--restore_path",
                type=str,
                default="",
                help="The save path for model")
      
      parser.add_argument(
                "--model_mode",
                type=str,
                default="train",
                help="The save path for model")
      FLAGS, unparsed = parser.parse_known_args()

      if FLAGS.model_mode == 'train':
            training()
      elif FLAGS.model_mode == 'eval':
            evaluation()
      elif FLAGS.model_mode == 'infer':
            inference()
      else:
            raise ValueError("model mode must be one of: 'train', 'eval','infer'.")
