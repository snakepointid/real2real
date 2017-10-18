from __future__ import print_function
import numpy as np
import sys
import re
from itertools import groupby
from operator import itemgetter
import cPickle as pickle
from real2real.preprocess.sentence_process import quick_sentence_segment 
from real2real.app.params import convClsParams

def LoadData():
	for line in sys.stdin:
		yield line.strip().split('\t')

def LoadTrainFeeds():
		reader = LoadData()
		title_batch,content_batch,target_batch=[],[],[]
		title_train,content_train,target_train=[],[],[]
		title_valid,content_valid,target_valid=[],[],[]
		title_testa,content_testa,target_testa=[],[],[]
		cache={'training':[]}
		for line in reader:
				rdv,target,title,content,flag=line
				title=title.split('|');content=content.split('|')
				if len(title)<4 or len(content)<20:
					continue
				title=title[:convClsParams.title_maxlen]+[0]*(convClsParams.title_maxlen-len(title))
				content=content[:convClsParams.content_maxlen]+[0]*(convClsParams.content_maxlen-len(content))

				if flag=='test':
						if abs(float(rdv))<convClsParams.test_rate:
								title_testa.append(title)
								content_testa.append(content)
								target_testa.append(target)		
						continue
						
				if abs(float(rdv))<convClsParams.test_rate:
						title_valid.append(title)
						content_valid.append(content)
						target_valid.append(target)
				else:
						title_batch.append(title)
		 				content_batch.append(content)
		 				target_batch.append(target)
		 				if abs(float(rdv))>(1-convClsParams.test_rate):
 								title_train.append(title)
								content_train.append(content)
								target_train.append(target)

				if len(title_batch)==convClsParams.batch_size:
						title_batch=np.array(title_batch,dtype=np.int64)
						content_batch=np.array(content_batch,dtype=np.int64)
						target_batch=np.array(target_batch,dtype=np.int32)
						train_cache=[title_batch,content_batch,target_batch]

						cache['training'].append(train_cache)
						title_batch,content_batch,target_batch = [],[],[]
 
		title_valid+=title_batch;content_valid+=content_batch;target_valid+=target_batch
 		title_valid=np.array(title_valid,dtype=np.int64)
 		content_valid =np.array(content_valid,dtype=np.int64)
 		target_valid=np.array(target_valid,dtype=np.int32)
		cache['valid']=[title_valid,content_valid,target_valid]

		title_train=np.array(title_train,dtype=np.int64)
 		content_train =np.array(content_train,dtype=np.int64)
 		target_train=np.array(target_train,dtype=np.int32)
		cache['train']=[title_train,content_train,target_train]

		title_testa=np.array(title_testa,dtype=np.int64)
 		content_testa =np.array(content_testa,dtype=np.int64)
 		target_testa=np.array(target_testa,dtype=np.int32)
		cache['testa']=[title_testa,content_testa,target_testa]

		return cache

def LoadPredictFeeds():
		content2code=pickle.load(open('/home/hdp-reader-content/shechanglue/titles/recallcontent2code.pkl','rb'))
		zh2code=pickle.load(open('/home/hdp-reader-content/shechanglue/titles/zh2code.pkl','rb'))
		zh2code['#NUMB#']=len(zh2code)+2
		zh2code['#ENG#']=len(zh2code)+2

		reader=LoadData()
		raw_batch,title_batch,content_batch=[],[],[]
		for line in reader:
				if len(line)!=3:
						continue
				url,recallcontent,title=line
				content=content2code.get(recallcontent,0)
				try:
						title=["%s"%zh2code.get(char,"1") for char in quick_sentence_segment(title.decode('utf-8'))]
						if len(title)<4:
								continue		
						title=title[:convClsParams.title_maxlen]+[0]*(convClsParams.title_maxlen-len(title))		
				except:
						continue
				raw='%s\t%s\t%s'%(url,recallcontent,title)
				raw_batch.append(raw)
				title_batch.append(title)
				content_batch.append(content)
				if len(title_batch)==convClsParams.batch_size:
						title_batch=np.array(title_batch,dtype=np.int64)
						content_batch=np.array(content_batch,dtype=np.int64)
						predict_cache=[raw_batch,title_batch,content_batch]
						raw_batch,title_batch,content_batch=[],[],[]
						yield predict_cache
																																							
		if len(title_batch):
        		title_batch=np.array(title_batch,dtype=np.int64)
        		content_batch=np.array(content_batch,dtype=np.int64)
        		predict_cache=[raw_batch,title_batch,content_batch]
        		yield predict_cache

if __name__ =="__main__":
	reader=LoadData()
	for line in reader:
		print (line)