from __future__ import print_function
import numpy as np
import sys
import re
from itertools import groupby
from operator import itemgetter
import cPickle as pickle
from real2real.preprocess.sentence_process import quick_sentence_segment 
from real2real.app.params import ctrRankModelParams

def LoadData():
	for line in sys.stdin:
		yield line.strip().split('\t')

def LoadTrainFeeds():
		reader = LoadData()
		source_batch,tag_batch,target_batch=[],[],[]
		source_train,tag_train,target_train=[],[],[]
		source_valid,tag_valid,target_valid=[],[],[]
		source_testa,tag_testa,target_testa=[],[],[]
		cache={'training':[]}
		for line in reader:
				if len(line)!=5:
					continue
				rdv,tag_code,source,target,flag=line
				source=source.split('|')
				if len(source)<4:
					continue
				source=source[:ctrRankModelParams.title_maxlen]+[0]*(ctrRankModelParams.title_maxlen-len(source))

				if flag=='test':
						if abs(float(rdv))<ctrRankModelParams.test_rate:
								source_testa.append(source)
								tag_testa.append(tag_code)
								target_testa.append(target)		
						continue
						
				if abs(float(rdv))<ctrRankModelParams.test_rate:
						source_valid.append(source)
						tag_valid.append(tag_code)
						target_valid.append(target)
				else:
						source_batch.append(source)
		 				tag_batch.append(tag_code)
		 				target_batch.append(target)
		 				if abs(float(rdv))>(1-ctrRankModelParams.test_rate):
 								source_train.append(source)
								tag_train.append(tag_code)
								target_train.append(target)

				if len(source_batch)==ctrRankModelParams.batch_size:
						source_batch=np.array(source_batch,dtype=np.int64)
						tag_batch=np.array(tag_batch,dtype=np.int64)
						target_batch=np.array(target_batch,dtype=np.float32)
						train_cache=[source_batch,tag_batch,target_batch]

						cache['training'].append(train_cache)
						source_batch,tag_batch,target_batch = [],[],[]
 
		source_valid+=source_batch;tag_valid+=tag_batch;target_valid+=target_batch
 		source_valid=np.array(source_valid,dtype=np.int64)
 		tag_valid =np.array(tag_valid,dtype=np.int64)
 		target_valid=np.array(target_valid,dtype=np.float32)
		cache['valid']=[source_valid,tag_valid,target_valid]

		source_train=np.array(source_train,dtype=np.int64)
 		tag_train =np.array(tag_train,dtype=np.int64)
 		target_train=np.array(target_train,dtype=np.float32)
		cache['train']=[source_train,tag_train,target_train]

		source_testa=np.array(source_testa,dtype=np.int64)
 		tag_testa =np.array(tag_testa,dtype=np.int64)
 		target_testa=np.array(target_testa,dtype=np.float32)
		cache['testa']=[source_testa,tag_testa,target_testa]

		return cache

def LoadPredictFeeds():
		tag2code=pickle.load(open('/home/hdp-reader-tag/shechanglue/sources/recalltag2code.pkl','rb'))
		zh2code=pickle.load(open('/home/hdp-reader-tag/shechanglue/sources/zh2code.pkl','rb'))
		zh2code['#NUMB#']=len(zh2code)+2
		zh2code['#ENG#']=len(zh2code)+2

		reader=LoadData()
		raw_batch,source_batch,tag_batch=[],[],[]
		for line in reader:
				if len(line)!=3:
						continue
				url,recalltag,title=line
				tag_code=tag2code.get(recalltag,0)
				try:
						source=["%s"%zh2code.get(char,"1") for char in quick_sentence_segment(title.decode('utf-8'))]
						if len(source)<4:
								continue		
						source=source[:ctrRankModelParams.title_maxlen]+[0]*(ctrRankModelParams.title_maxlen-len(source))		
				except:
						continue
				raw='%s\t%s\t%s'%(url,recalltag,title)
				raw_batch.append(raw)
				source_batch.append(source)
				tag_batch.append(tag_code)
				if len(source_batch)==ctrRankModelParams.batch_size:
						source_batch=np.array(source_batch,dtype=np.int64)
						tag_batch=np.array(tag_batch,dtype=np.int64)
						predict_cache=[raw_batch,source_batch,tag_batch]
						raw_batch,source_batch,tag_batch=[],[],[]
						yield predict_cache
																																							
		if len(source_batch):
        		source_batch=np.array(source_batch,dtype=np.int64)
        		tag_batch=np.array(tag_batch,dtype=np.int64)
        		predict_cache=[raw_batch,source_batch,tag_batch]
        		yield predict_cache

if __name__ =="__main__":
	cache=LoadTrainFeeds()
	print ("train nums is %s\ttest nums is %s"%(len(cache['training'])*ctrRankModelParamss.batch_size,len(cache['valid'][1])))
