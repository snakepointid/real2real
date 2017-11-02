from __future__ import print_function
import numpy as np
import sys
import re
from itertools import groupby
from operator import itemgetter
import cPickle as pickle
from real2real.preprocess.sentence_process import quick_sentence_segment 
from real2real.app.params import languageModelParams

def LoadData():
	for line in sys.stdin:
		yield line.strip().split('\t')

def LoadTrainFeeds():
		reader = LoadData()
		source_batch=[]
		source_train=[]
		source_valid=[]
		cache={'training':[]}
		for line in reader:
				if len(line)!=4:
						continue
				_,source,target,rdv=line
				source=source.split('|') 
				if len(source)<4 or len(target)<4:
						continue
				source=source[:languageModelParams.source_maxlen]+[0]*(languageModelParams.source_maxlen-len(source))
						
				if abs(float(rdv))<languageModelParams.test_rate:
						source_valid.append(source)
				else:
						source_batch.append(source)
		 				if abs(float(rdv))>(1-languageModelParams.test_rate):
 								source_train.append(source)

				if len(source_batch)==languageModelParams.batch_size:
						source_batch=np.array(source_batch,dtype=np.int64)
						cache['training'].append(source_batch)
						source_batch = []
 
		source_valid+=source_batch
 		source_valid=np.array(source_valid,dtype=np.int64)
		cache['valid']=source_valid
		source_train=np.array(source_train,dtype=np.int64)
		cache['train']=source_train
		return cache

if __name__ =="__main__":
	cache=LoadTrainFeeds()
	print (cache['train'][0])
	print ("train nums is %s\ttest nums is %s"%(len(cache['train']),len(cache['valid'][1])))
