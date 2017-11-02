from __future__ import print_function
import tensorflow as tf
import sys
import time
import argparse
import os
sys.path.insert(0,"..")
sys.path.append(os.getcwd())
from real2real.models.seq2seq import LanguageModel
from real2real.app.params import appParams
from real2real.preprocess.languageModel_feeds import *
from real2real.utils.info_layout import *

def training():
        gpu_options = tf.GPUOptions(allow_growth = True)

        model = LanguageModel(is_training=True)

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
                        for source_batch in cache['training']:
                                _,gs=sess.run([model.train_op,model.global_step],feed_dict={
                                                                model.source_code:source_batch,
                                                                model.is_dropout:True})
     
                        source_batch = cache['train'] 
                        train_loss,train_acc = sess.run([model.mean_loss,model.acc],feed_dict={
                                                                model.source_code:source_batch,                                            
                                                                model.is_dropout:False})

                        train_num=len(source_batch)
                        source_batch = cache['valid']   
                        valid_loss,valid_acc = sess.run([model.mean_loss,model.acc],feed_dict={
                                                                model.source_code:source_batch,                                            
                                                                model.is_dropout:False})
                        valid_num=len(source_batch)

                        print("Iteration:%s\ttrain num:%s\ttrain acc:%s\ttrain loss:%s\tvalid num:%s\tvalid acc:%s\tvalid loss:%s"\
                        %(gs,train_num,train_acc,train_loss,valid_num,valid_acc,valid_loss))  

                        endTime = time.time()
                        if endTime-startTime>3600:
                                print ("save the whole model")
                                model.global_saver.save(sess,FLAGS.save_path+"/global_model")
                                startTime = endTime
                print ("save the whole model")
                model.global_saver.save(sess,FLAGS.save_path+"/global_model")
 
def evaluation():
        label2code=pickle.load(open('/home/hdp-reader-tag/shechanglue/sources/label2code.pkl','rb'))
        code2label= dict(zip(label2code.values(), label2code.keys()))
        gpu_options = tf.GPUOptions(allow_growth = True)
        model = LanguageModel(is_training=False)
        with tf.Session(graph = model.graph,config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True, log_device_placement = False)) as sess:
                model.global_saver.restore(sess,FLAGS.restore_path+"/global_model")
                cache = LoadEvalFeeds()
                acc = [] 
                for raw_batch,source_batch in cache:
                        testa_acc,testa_pred,probs = sess.run([model.acc,model.preds,model.probs],feed_dict={
                                                                model.source_code:source_batch,                                            
                                                                model.is_dropout:False}) 
                        acc.append(testa_acc)
                        for idx,raw in enumerate(raw_batch):
                                if testa_pred[idx]![idx]:
                                        print("%s\t%s\tacc:%s\t%s\t%s"%(raw,probs[idx][idx]]\
                                            ,np.mean(acc),code2label.get(testa_pred[idx],'UNK'),probs[idx][testa_pred[idx]]))
                                        print('-'*150)
def inference():
        gpu_options = tf.GPUOptions(allow_growth = True)
        model = LanguageModel(is_training=False)
        with tf.Session(graph = model.graph,config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True, log_device_placement = False)) as sess:
                model.global_saver.restore(sess,FLAGS.restore_path+"/global_model")
                cache = LoadPredictFeeds()
    
                for raw_batch,title_source_batch,tag_batch in cache:
                        probs=sess.run(model.logits,feed_dict={
                                                        model.source:source_batch ,
                                                        model.tag:tag_batch ,
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
