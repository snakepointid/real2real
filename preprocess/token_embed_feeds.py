from __future__ import print_function
import numpy as np
import sys
import re
from itertools import groupby
from operator import itemgetter
import cPickle as pickle
from real2real.app.params import tokenEmbedModelParams

def LoadData():
	for line in sys.stdin:
		yield line.strip().split('\t')

def LoadTrainFeeds():
		reader = LoadData()
		pair_batch,target_batch=[],[]
		pair_train,target_train=[],[]
		pair_valid,target_valid=[],[]
		cache={'training':[]}
		for line in reader:
				if len(line)!=4:
						continue
				rdv,token_a,token_b,target = line
				if abs(float(rdv))<tokenEmbedModelParams.test_rate:
						pair_valid.append([token_a,token_b])
						target_valid.append(target)
				else:
						pair_batch.append([token_a,token_b])
		 				target_batch.append(target)
		 				if abs(float(rdv))>(1-tokenEmbedModelParams.test_rate):
 								pair_train.append([token_a,token_b])
								target_train.append(target)

				if len(pair_batch)==tokenEmbedModelParams.batch_size:
						pair_batch=np.array(pair_batch,dtype=np.int64)
						target_batch=np.array(target_batch,dtype=np.int32)
						train_cache=[pair_batch,target_batch]
						cache['training'].append(train_cache)
						pair_batch,target_batch = [],[]
 
		pair_valid+=pair_batch;target_valid+=target_batch
 		pair_valid=np.array(pair_valid,dtype=np.int64)
 		target_valid=np.array(target_valid,dtype=np.int32)
		cache['valid']=[pair_valid,target_valid]

		pair_train=np.array(pair_train,dtype=np.int64)
 		target_train=np.array(target_train,dtype=np.int32)
		cache['train']=[pair_train,target_train]

		return cache


if __name__ =="__main__":
	reader=LoadData()
	for line in reader:
		print (line)
