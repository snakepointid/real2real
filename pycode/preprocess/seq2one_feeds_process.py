from __future__ import print_function
import numpy as np
import sys
import re
from itertools import groupby
from operator import itemgetter
import cPickle as pickle
from preprocess.sentence_process import sentenceSeg 
from app.params import convRankParams
sys.path.insert(0,'..')

def LoadData():
	for line in sys.stdin:
		yield line.strip().split('\t')

def LoadTrainFeeds():
	reader = LoadData()
	text_code_batch,tag_code_batch,ctr_batch = [],[],[]
	text_code_test,tag_code_test,ctr_test = [],[],[]
	cache = []
	for line in reader:
		#try:
			_,_,tagcode,textcode,ctr,rdv = line
			textcode = textcode.split('|')
			textcode = textcode[:convRankParams.source_maxlen]+[0]*(convRankParams.source_maxlen-len(textcode))
			if float(rdv)<-(1-convRankParams.test_rate):
				text_code_test.append(textcode)
				tag_code_test.append(tagcode)
				ctr_test.append(ctr)
			else:
				text_code_batch.append(textcode)
        	                tag_code_batch.append(tagcode)
        	                ctr_batch.append(ctr)
	
			if len(text_code_batch)==convRankParams.batch_size:
				text_code_batch = np.array(text_code_batch,dtype=np.int64)
				tag_code_batch  = np.array(tag_code_batch,dtype=np.int64)
				ctr_batch       = np.array(ctr_batch,dtype=np.float32)
				train_cache = [text_code_batch,tag_code_batch,ctr_batch]
				#yield 'train',train_cache
				cache.append(['train',train_cache])
				text_code_batch,tag_code_batch,ctr_batch = [],[],[]
		#except:
			#print '\t'.join(line)
		#	continue

 	text_code_test = np.array(text_code_test,dtype=np.int64)
 	tag_code_test  = np.array(tag_code_test,dtype=np.int64)
 	ctr_test       = np.array(ctr_test,dtype=np.float32)
	test_cache = [text_code_test,tag_code_test,ctr_test]
	cache.append(['test',test_cache])
	#yield 'test',test_cache
	return cache

def LoadPredictFeeds():
	tag2code = pickle.load(open('/home/hdp-reader-tag/shechanglue/sources/recalltag2code.pkl','rb'))
	zh2code = pickle.load(open('/home/hdp-reader-tag/shechanglue/sources/zh2code.pkl','rb'))
	zh2code['#NUMB#']=len(zh2code)+2
	zh2code['#ENG#']=len(zh2code)+2
	reader = LoadData()
	raw_batch,text_code_batch,tag_code_batch=[],[],[]
        for line in reader:
		if len(line)!=3:
			continue
		url,recalltag,title = line
                tagcode = tag2code.get(recalltag,0)
		try:
                	textcode = ["%s"%zh2code.get(char,"1") for char in sentenceSeg(title.decode('utf-8'))]
			if len(textcode)<4:
				continue
			textcode = textcode[:convRankParams.textLen]+[0]*(convRankParams.textLen-len(textcode))	
		except:
			continue
		raw = '%s\t%s\t%s'%(url,recalltag,title)
		raw_batch.append(raw)
		text_code_batch.append(textcode)
		tag_code_batch.append(tagcode)
                if len(text_code_batch)==convRankParams.batch_size:
			text_code_batch = np.array(text_code_batch,dtype=np.int64)
                        tag_code_batch  = np.array(tag_code_batch,dtype=np.int64)
			predict_cache   = [raw_batch,text_code_batch,tag_code_batch]
			raw_batch,text_code_batch,tag_code_batch=[],[],[]
			yield predict_cache
	if len(text_code_batch):
        	text_code_batch = np.array(text_code_batch,dtype=np.int64)
        	tag_code_batch  = np.array(tag_code_batch,dtype=np.int64)
        	predict_cache   = [raw_batch,text_code_batch,tag_code_batch]
        	yield predict_cache

if __name__ =="__main__":
	train_cache,test_cache = LoadTrainFeeds()
	print ("train nums is %s\ttest nums is %s"%(len(train_cache)*convRankParams.batch_size,len(test_cache[1])))
