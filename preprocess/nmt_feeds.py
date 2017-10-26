from __future__ import print_function
import numpy as np
import sys
import re
from itertools import groupby
from operator import itemgetter
import cPickle as pickle
from real2real.preprocess.sentence_process import quick_sentence_segment 
from real2real.app.params import nmtModelParams

def LoadData():
	for line in sys.stdin:
		yield line.strip().split('\t')

def LoadTrainFeeds():
		reader = LoadData()
		source_batch,target_batch=[],[]
		source_train,target_train=[],[]
		source_valid,target_valid=[],[]
		cache={'training':[]}
		for line in reader:
				if len(line)!=4:
					continue
				_,source,target,rdv=line
				source=source.split('|');target=target.split('|')
				if len(source)<4 or len(target)<4:
					continue
				source=source[:nmtModelParams.source_maxlen]+[0]*(nmtModelParams.source_maxlen-len(source))
				target=target[:nmtModelParams.target_maxlen]+[0]*(nmtModelParams.target_maxlen-len(target))
						
				if abs(float(rdv))<nmtModelParams.test_rate:
						source_valid.append(source)
						target_valid.append(target)
				else:
						source_batch.append(source)
		 				target_batch.append(target)
		 				if abs(float(rdv))>(1-nmtModelParams.test_rate):
 								source_train.append(source)
								target_train.append(target)

				if len(source_batch)==nmtModelParams.batch_size:
						source_batch=np.array(source_batch,dtype=np.int64)
						target_batch=np.array(target_batch,dtype=np.int64)
						train_cache=[source_batch,,target_batch]

						cache['training'].append(train_cache)
						source_batch,target_batch = [],[]
 
		source_valid+=source_batch;+=;target_valid+=target_batch
 		source_valid=np.array(source_valid,dtype=np.int64)
 		target_valid=np.array(target_valid,dtype=np.int64)
		cache['valid']=[source_valid,,target_valid]

		source_train=np.array(source_train,dtype=np.int64)
 		target_train=np.array(target_train,dtype=np.int64)
		cache['train']=[source_train,,target_train]
		
		return cache

def LoadPredictFeeds():
		tag2code=pickle.load(open('/home/hdp-reader-tag/shechanglue/sources/recalltag2code.pkl','rb'))
		zh2code=pickle.load(open('/home/hdp-reader-tag/shechanglue/sources/zh2code.pkl','rb'))
		zh2code['#NUMB#']=len(zh2code)+2
		zh2code['#ENG#']=len(zh2code)+2

		reader=LoadData()
		raw_batch,source_batch,=[],[],[]
		for line in reader:
				if len(line)!=3:
						continue
				url,recalltag,title=line
				tag_code=tag2code.get(recalltag,0)
				try:
						source=["%s"%zh2code.get(char,"1") for char in quick_sentence_segment(title.decode('utf-8'))]
						if len(source)<4:
								continue		
						source=source[:nmtModelParams.source_maxlen]+[0]*(nmtModelParams.source_maxlen-len(source))		
				except:
						continue
				raw='%s\t%s\t%s'%(url,recalltag,title)
				raw_batch.append(raw)
				source_batch.append(source)
				.append(tag_code)
				if len(source_batch)==nmtModelParams.batch_size:
						source_batch=np.array(source_batch,dtype=np.int64)
						=np.array(,dtype=np.int64)
						predict_cache=[raw_batch,source_batch,]
						raw_batch,source_batch,=[],[],[]
						yield predict_cache
																																							
		if len(source_batch):
        		source_batch=np.array(source_batch,dtype=np.int64)
        		=np.array(,dtype=np.int64)
        		predict_cache=[raw_batch,source_batch,]
        		yield predict_cache

if __name__ =="__main__":
	cache=LoadTrainFeeds()
	print ("train nums is %s\ttest nums is %s"%(len(cache['training'])*nmtModelParamss.batch_size,len(cache['valid'][1])))
