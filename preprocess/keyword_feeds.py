from __future__ import print_function
import numpy as np
import sys
import re
from itertools import groupby
from operator import itemgetter
import cPickle as pickle
from real2real.preprocess.sentence_process import quick_sentence_segment 
from real2real.app.params import keywordModelParams
stop_words = [1,81946]
def LoadData():
		for line in sys.stdin:
				yield line.strip().split('\t')

def LoadTrainFeeds():
		reader = LoadData()
		source_batch,label_batch=[],[]
		source_train,label_train=[],[]
		source_valid,label_valid=[],[]
		cache={'training':[]}
		for line in reader:
				if len(line)!=3:
						continue
				rdv,source,label_code=line
				raw,source = source.split("")
				source=list(set([code for code in source.split('|') if int(code) not in stop_words]))
				
				if len(source)<3:
						continue
				source=source[:keywordModelParams.title_maxlen]+[0]*(keywordModelParams.title_maxlen-len(source))
						
				if abs(float(rdv))<keywordModelParams.test_rate:
						source_valid.append(source)
						label_valid.append(label_code)
				else:
						source_batch.append(source)
						label_batch.append(label_code)
						if abs(float(rdv))>(1-keywordModelParams.test_rate):
								source_train.append(source)
								label_train.append(label_code)

				if len(source_batch)==keywordModelParams.batch_size:
						source_batch=np.array(source_batch,dtype=np.int64)
						label_batch=np.array(label_batch,dtype=np.int64)
						train_cache=[source_batch,label_batch]
						cache['training'].append(train_cache)
						source_batch,label_batch = [],[]

		source_valid+=source_batch;label_valid+=label_batch
		source_valid=np.array(source_valid,dtype=np.int64)
		label_valid =np.array(label_valid,dtype=np.int64)
		cache['valid']=[source_valid,label_valid]

		source_train=np.array(source_train,dtype=np.int64)
		label_train =np.array(label_train,dtype=np.int64)
		cache['train']=[source_train,label_train]

		return cache

def LoadEvalFeeds():
		reader = LoadData()
		raw_batch,source_batch,label_batch=[],[],[]
		for line in reader:
				if len(line)!=3:
						continue
				rdv,source,label_code=line
				raw,source = source.split("")
				source=list(set([code for code in source.split('|') if int(code) not in stop_words]))
				if len(source)<3:
						continue
				source=source[:keywordModelParams.title_maxlen]+[0]*(keywordModelParams.title_maxlen-len(source))
				raw_batch.append(raw.replace("|",""))		
				source_batch.append(source)
				label_batch.append(label_code)
				if len(source_batch)==keywordModelParams.batch_size:
						source_batch=np.array(source_batch,dtype=np.int64)
						label_batch=np.array(label_batch,dtype=np.int64)
						cache=[raw_batch,source_batch,label_batch]
						raw_batch,source_batch,label_batch=[],[],[]
						yield cache
						
		if len(source_batch):
				source_batch=np.array(source_batch,dtype=np.int64)
				label_batch=np.array(label_batch,dtype=np.int64)
				cache=[raw_batch,source_batch,label_batch]
				yield cache

def LoadInferFeeds():
		reader = LoadData()
		raw_batch,source_batch,label_batch=[],[],[]
		chi2code   = pickle.load(open("/home/hdp-reader-tag/shechanglue/sources/entity_encode/chi2code.pkl","rb"))
		label2code = pickle.load(open("/home/hdp-reader-tag/shechanglue/sources/entity_encode/label2code.pkl","rb"))

		for line in reader:
				if len(line)==4:
					token,label,_,_=line 
				elif len(line)==2:
					label,token=line
				else:
					continue
 				token_code = chi2code.get(token,0)
 				label_code = label2code.get(label,0)
				if token_code==0 or label_code==0:
						continue
				raw = "%s\t%s"%(token,label)
				source=[token_code]*30
				raw_batch.append(raw)		
				source_batch.append(source)
				label_batch.append(label_code)
				if len(source_batch)==keywordModelParams.batch_size:
						source_batch=np.array(source_batch,dtype=np.int64)
						label_batch=np.array(label_batch,dtype=np.int64)
						cache=[raw_batch,source_batch,label_batch]
						raw_batch,source_batch,label_batch=[],[],[]
						yield cache
						
		if len(source_batch):
				source_batch=np.array(source_batch,dtype=np.int64)
				label_batch=np.array(label_batch,dtype=np.int64)
				cache=[raw_batch,source_batch,label_batch]
				yield cache

if __name__ =="__main__":
		cache=LoadTrainFeeds()
		print ("train nums is %s\ttest nums is %s"%(len(cache['training'])*keywordModelParamss.batch_size,len(cache['valid'][1])))



















