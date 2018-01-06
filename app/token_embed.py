from __future__ import print_function
import tensorflow as tf
import sys
import time
import argparse
import os
sys.path.insert(0,"..")
sys.path.append(os.getcwd())
from real2real.models.entity_embed import TokenEmbed
from real2real.app.params import appParams,tokenEmbedModelParams
from real2real.preprocess.token_embed_feeds import *
from real2real.utils.info_layout import *

language = tokenEmbedModelParams.language
def training():
        gpu_options = tf.GPUOptions(allow_growth = True)
        model = TokenEmbed(is_training=True)

        with tf.Session(graph = model.graph,config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True, log_device_placement = False)) as sess:
                try:
                        sess.run(model.init_op)
                        model.token_embed_savers[language].restore(sess,FLAGS.restore_path+"/%s"%language)
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
                        train_num = len(pair_batch)
                        train_loss = sess.run(model.mean_loss,feed_dict={
                                                                model.pair:pair_batch,      
                                                                model.target:label_batch,
                                                                model.is_dropout:False})

                        pair_batch,label_batch = cache['valid']   
                        test_num = len(pair_batch)
                        test_loss = sess.run(model.mean_loss,feed_dict={
                                                                model.pair:pair_batch,      
                                                                model.target:label_batch,
                                                                model.is_dropout:False})
                        print("Iteration:%s\ttrain num:%s\ttrain loss:%s\t\ttest num:%s\ttest loss:%s"%(gs,train_num,train_loss,test_num,test_loss))
                        if epoch%10==0:
                                print ("save the whole model")
                                model.token_embed_savers[language].save(sess,FLAGS.save_path+"/%s"%language)
         
                print ("save the whole model")
                model.token_embed_savers[language].save(sess,FLAGS.save_path+"/%s"%language)

def evaluation():  
        chi2code = pickle.load(open("/home/hdp-reader-tag/shechanglue/sources/entity_encode/chi2code.pkl","rb"))
        chi2code['UNK']=1
        code2chi = dict(zip(chi2code.values(), chi2code.keys()))

        model = TokenEmbed(is_training=False)
        gpu_options = tf.GPUOptions(allow_growth = True)
        with tf.Session(graph = model.graph,config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True, log_device_placement = False)) as sess:
                model.token_embed_savers[language].restore(sess,FLAGS.restore_path+"/%s"%language)
       		model.infer()
                query_tokens = np.arange(tokenEmbedModelParams.source_vocab_size)
                batch_size = tokenEmbedModelParams.source_vocab_size/appParams.num_epochs
                for batch in range(appParams.num_epochs): 
                        query_batch = query_tokens[batch_size*batch:batch_size*(batch+1)]          
                        most_k_similar_tokens = sess.run(model.most_k_similar,feed_dict={
                                                                        model.query:query_batch,  
                                                                        model.is_dropout:False})
                        for idx,token_code in enumerate(query_batch):
                                   most_k_similar = "->".join([code2chi.get(code,'PAD')for code in most_k_similar_tokens[idx]])
                                   print("%s\t:\t%s"%(code2chi.get(token_code,'PAD'),most_k_similar))


 
 
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
        else:
		raise ValueError("model mode must be 'train'or'eval'")
