from __future__ import print_function
import numpy as np
import sys
import re
from itertools import groupby
from operator import itemgetter
import cPickle as pickle
from real2real.preprocess.sentence_process import quick_sentence_segment 
from real2real.app.params import LDAModelParams 
import random
def LoadData():
		for line in sys.stdin:
				yield line.strip().split('\t')

def LoadTrainFeeds():
		reader = LoadData()
		user_batch,tag_batch,label_batch=[],[],[]
		user_train,tag_train,label_train=[],[],[]
		user_valid,tag_valid,label_valid=[],[],[]
		cache={'training':[]}
		for line in reader:
				if len(line)!=3:
						continue
				user,tag,flag=line
				rdv= random.uniform(0,1)
				if rdv<LDAModelParams.test_rate:
						user_valid.append(user)
						tag_valid.append(tag)
						label_valid.append(flag)

				else:
						user_batch.append(user)
						tag_batch.append(tag)
						label_batch.append(flag)
						if rdv>(1-LDAModelParams.test_rate):
								user_train.append(user)
								tag_train.append(tag)
								label_train.append(flag)

				if len(user_batch)==LDAModelParams.batch_size:
						user_batch=np.array(user_batch,dtype=np.int64)
						tag_batch=np.array(tag_batch,dtype=np.int64)
						label_batch = np.array(label_batch,dtype=np.float32)
						train_cache=[user_batch,tag_batch,label_batch]
						cache['training'].append(train_cache)
						user_batch,tag_batch,label_batch = [],[],[]

		user_valid+=user_batch;tag_valid+=tag_batch;label_valid+=label_batch;
		user_valid=np.array(user_valid,dtype=np.int64)
		tag_valid =np.array(tag_valid,dtype=np.int64)
		label_valid = np.array(label_valid,dtype=np.float32)
		cache['valid']=[user_valid,tag_valid,label_valid]

		user_train=np.array(user_train,dtype=np.int64)
		tag_train =np.array(tag_train,dtype=np.int64)
		label_train = np.array(label_train,dtype=np.float32)
		cache['train']=[user_train,tag_train,label_train]

		return cache

 

if __name__ =="__main__":
		cache=LoadTrainFeeds()
		print ("train nums is %s\ttest nums is %s"%(len(cache['training'])*LDAModelParams.batch_size,len(cache['valid'][1])))



















