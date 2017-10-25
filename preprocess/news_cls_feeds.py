from __future__ import print_function
import numpy as np
import sys
import re
from itertools import groupby
from operator import itemgetter
import cPickle as pickle
from real2real.preprocess.sentence_process import quick_sentence_segment 
from real2real.app.params import newsClsModelParams

def LoadData():
	for line in sys.stdin:
		yield line.strip().split('\t')

def LoadTrainFeeds():
		reader = LoadData()
		title_batch,content_batch,target_batch=[],[],[]
		title_train,content_train,target_train=[],[],[]
		title_valid,content_valid,target_valid=[],[],[]
		cache={'training':[]}
		for line in reader:
				rdv,target,title,content,flag=line
				title=title.split('|');content=content.split('|')
				if len(title)<4 or len(content)<20:
					continue
				title=title[:newsClsModelParams.title_maxlen]+[0]*(newsClsModelParams.title_maxlen-len(title))
				content=content[:newsClsModelParams.content_maxlen]+[0]*(newsClsModelParams.content_maxlen-len(content))
						
				if abs(float(rdv))<newsClsModelParams.test_rate:
						title_valid.append(title)
						content_valid.append(content)
						target_valid.append(target)
				else:
						title_batch.append(title)
		 				content_batch.append(content)
		 				target_batch.append(target)
		 				if abs(float(rdv))>(1-newsClsModelParams.test_rate):
 								title_train.append(title)
								content_train.append(content)
								target_train.append(target)

				if len(title_batch)==newsClsModelParams.batch_size:
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

		return cache

def LoadEvalFeeds():
		content2code=pickle.load(open('/home/hdp-reader-tag/shechanglue/titles/recalltag2code.pkl','rb'))
		zh2code=pickle.load(open('/home/hdp-reader-tag/shechanglue/titles/zh2code.pkl','rb'))
		label2code=pickle.load(open('/home/hdp-reader-tag/shechanglue/titles/label2code.pkl','rb'))
		zh2code['#NUMB#']=len(zh2code)+2
		zh2code['#ENG#']=len(zh2code)+2
		reader=LoadData()
		raw_batch,title_batch,content_batch,target_batch=[],[],[],[]
		for line in reader:
				line = re.sub(r'<\w+>','',line)
                line = re.sub(r'<\/\w+>','\t',line)
                sep = line.split('\t')
                if len(sep)!=5:
                        continue
                label,url,title,content,_ = sep
                raw='%s\t%s\t%s'%(title,content,label)
                title = ["%s"%zh2code.get(char,"1") for char in quick_sentence_segment(title.decode('utf-8'))]
                content = ["%s"%zh2code.get(char,"1") for char in quick_sentence_segment(content.decode('utf-8'))]
                target = label2code.get(label,0)

				if len(title)<4 or len(content)<20:
					continue		
				title=title[:newsClsModelParams.title_maxlen]+[0]*(newsClsModelParams.title_maxlen-len(title))
				content=content[:newsClsModelParams.content_maxlen]+[0]*(newsClsModelParams.content_maxlen-len(content))
						
				title_batch.append(title)
				content_batch.append(content)
				target_batch.append(target)
				raw_batch.append(raw)
				if len(title_valid)==100:
						title_batch=np.array(title_batch,dtype=np.int64)
						content_batch=np.array(content_batch,dtype=np.int64)
						target_batch=np.array(target_batch,dtype=np.int32)
						yield [title_batch,content_batch,target_batch,raw_batch]
						raw_batch,title_batch,content_batch,target_batch=[],[],[],[]
			 																																					
if __name__ =="__main__":
	reader=LoadData()
	for line in reader:
		print (line)
