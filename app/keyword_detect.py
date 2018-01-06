from __future__ import print_function
import tensorflow as tf
import sys
import time
import argparse
import os
sys.path.insert(0,"..")
sys.path.append(os.getcwd())
from real2real.models.seq2one import keywordModel
from real2real.app.params import baseModelParams
from real2real.preprocess.keyword_feeds import *
from real2real.utils.info_layout import *
from real2real.utils.metrics import regression_model_eval

def training():
      gpu_options = tf.GPUOptions(allow_growth = True)
      model = keywordModel(is_training=True)
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
            old_acc = 0
            #compute the initial pearson coef
            for epoch in range(baseModelParams.num_epochs):
                  for text_code_batch,label_code_batch in cache['training']:
    
                        _,gs=sess.run([model.train_op,model.global_step],feed_dict={
                                                            model.title_source:text_code_batch ,
                                                            model.target:label_code_batch,
                                                            model.is_dropout:True})
	 
                  text_code_batch,label_code_batch = cache['train']	
                  train_num = len(label_code_batch)
                  train_acc = sess.run(model.acc,feed_dict={
                                                            model.title_source:text_code_batch, 
                                                            model.target:label_code_batch,                                           
                                                            model.is_dropout:False})
                  text_code_batch,label_code_batch = cache['valid']   
                  test_num = len(label_code_batch)
                  test_acc = sess.run(model.acc,feed_dict={
                                                            model.title_source:text_code_batch,
                                                            model.target:label_code_batch,                                            
                                                            model.is_dropout:False})                  
                  print("iteration:%s\ttrain num:%s\ttrain acc:%s\ttest num:%s\ttest acc:%s"%(gs,train_num,train_acc,test_num,test_acc))
                  endTime = time.time()
                  if endTime-startTime>1000:
                        print ("save the whole model")
                        model.global_saver.save(sess,FLAGS.save_path+"/global_model")
                        startTime = endTime

            print ("save the whole model")
            model.global_saver.save(sess,FLAGS.save_path+"/global_model")
 
def inference():
      chi2code = pickle.load(open("/home/hdp-reader-tag/shechanglue/sources/entity_encode/chi2code.pkl","rb"))
      label2code = pickle.load(open("/home/hdp-reader-tag/shechanglue/sources/entity_encode/label2code.pkl","rb"))
      chi2code['UNK']=1
      label2code['UNK']=0
      code2chi = dict(zip(chi2code.values(), chi2code.keys()))
      code2label = dict(zip(label2code.values(), label2code.keys()))

      gpu_options = tf.GPUOptions(allow_growth = True)
      model = keywordModel(is_training=False)
      with tf.Session(graph = model.graph,config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True, log_device_placement = False)) as sess:
            model.global_saver.restore(sess,FLAGS.restore_path+"/global_model")

            cache = LoadInferFeeds()

            for raw_batch,source_batch,label_batch in cache:
                    weights_score = sess.run(model.weights_score,feed_dict={
                                                            model.title_source:source_batch,
                                                            model.target:label_batch,                                           
                                                            model.is_dropout:False})
                    for idx,raw in enumerate(raw_batch):
                          print("%s\t%s"%(raw,weights_score[idx][0]))
             

def evaluation():
      chi2code = pickle.load(open("/home/hdp-reader-tag/shechanglue/sources/entity_encode/chi2code.pkl","rb"))
      label2code = pickle.load(open("/home/hdp-reader-tag/shechanglue/sources/entity_encode/label2code.pkl","rb"))
      chi2code['UNK']=1
      label2code['UNK']=0
      code2chi = dict(zip(chi2code.values(), chi2code.keys()))
      code2label = dict(zip(label2code.values(), label2code.keys()))
      
      gpu_options = tf.GPUOptions(allow_growth = True)
      model = keywordModel(is_training=False)
      with tf.Session(graph = model.graph,config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True, log_device_placement = False)) as sess:
            model.global_saver.restore(sess,FLAGS.restore_path+"/global_model")
            cache = LoadEvalFeeds()
    
            for raw_batch,title_source_batch,label_batch in cache:
                  word_weights=sess.run(model.weights,feed_dict={
                                                            model.title_source:title_source_batch ,
						                                                model.target:label_batch,
                                                            model.is_dropout:False}) 

                  for idx,label in enumerate(label_batch):
                        label = code2label[label]
                        raw = raw_batch[idx]
                        title = title_source_batch[idx]
                        weight = word_weights[idx]
                        ret = {}
                        for idd,token in enumerate(title):
                              token = code2chi.get(token,0)
                              if token:
                                    ret[token] = round(weight[idd],3)
                        ret= sorted(ret.items(), key=lambda d:d[1], reverse = True)
                        prt = ""
                        for token,weight in ret:
                              if weight<0.1:
                                    continue
                              prt+="%s_%s|"%(token,weight)
                        print ("%s\t%s\t%s"%(label,raw,prt[:-1]))   

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
