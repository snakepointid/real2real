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
						train_cache=[source_batch,target_batch]

						cache['training'].append(train_cache)
						source_batch,target_batch = [],[]
 
		source_valid+=source_batch;target_valid+=target_batch
 		source_valid=np.array(source_valid,dtype=np.int64)
 		target_valid=np.array(target_valid,dtype=np.int64)
		cache['valid']=[source_valid,target_valid]

		source_train=np.array(source_train,dtype=np.int64)
 		target_train=np.array(target_train,dtype=np.int64)
		cache['train']=[source_train,target_train]
		
		return cache

if __name__ =="__main__":
	cache=LoadTrainFeeds()
	print (cache['train'][0])
	print ("train nums is %s\ttest nums is %s"%(len(cache['train']),len(cache['valid'][1])))
