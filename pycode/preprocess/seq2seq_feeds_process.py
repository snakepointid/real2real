from __future__ import print_function
import numpy as np
import sys
import re
from itertools import groupby
from operator import itemgetter
import cPickle as pickle
from preprocess.sentence_process import sentenceSeg 
from app.params import nmtParams
sys.path.insert(0,'..')

def LoadData():
	for line in sys.stdin:
		yield line.strip().split('\t')

def LoadTrainFeeds():
		reader = LoadData()
		source_batch,target_batch = [],[]
		source_train,target_train = [],[]
		source_valid,target_valid = [],[]
		source_testa,target_testa = [],[]
		cache = {'training':[]}
		for source,target,rdv in reader:
				source = source.split('|')
				source = source[:nmtParams.source_maxlen]+[0]*(nmtParams.source_maxlen-len(source))
				target = target.split('|')
				target = target[:nmtParams.target_maxlen]+[0]*(nmtParams.target_maxlen-len(target))

				if abs(float(rdv))<nmtParams.test_rate:
						source_valid.append(source)
						target_valid.append(target)
				else:

						source_batch.append(source)
		 				target_batch.append(target)
		 				if abs(float(rdv))>(1-nmtParams.test_rate):
 								source_train.append(source)
								target_train.append(target)

				if len(source_batch)==nmtParams.batch_size:
						source_batch = np.array(source_batch,dtype=np.int64)
						target_batch  = np.array(target_batch,dtype=np.int64)
						train_cache = [source_batch,target_batch]

						cache['training'].append(train_cache)
						source_batch,target_batch = [],[]
 
		source_valid+=source_batch;target_valid+=target_batch
 		source_valid = np.array(source_valid,dtype=np.int64)
 		target_valid = np.array(target_valid,dtype=np.int64)
		cache['valid'] = [source_valid,target_valid]

		source_train = np.array(source_train,dtype=np.int64)
 		target_train = np.array(target_train,dtype=np.int64)
		cache['train'] = [source_train,target_train]

		return cache

if __name__ =="__main__":
	cache = LoadTrainFeeds()
	print ("train nums is %s\ttest nums is %s"%(len(cache['training'])*nmtParams.batch_size,len(cache['valid'][1])))
