#! ./python27.tar.gz/python27/bin/python27
#-*- coding:utf-8 -*-
import sys
import re
import cPickle as pickle
import random

old_recalltag_count = pickle.load(open("recalltag_count.pkl","rb"));recalltag_count={}
recalltag2code = pickle.load(open("/home/hdp-reader-tag/shechanglue/sources/recalltag2code.pkl","rb"))
newtag = pickle.load(open("new_tag.pkl","rb"))

#get encoded tag's count
for recalltag in recalltag2code:
                recalltag_count[recalltag] = old_recalltag_count.get(recalltag,0)
#shrink the recalltag_count and newtag
for recalltag in newtag.keys():
                if recalltag_count.get(recalltag,-1)!=-1:
                                recalltag_count.pop(recalltag)
                                newtag.pop(recalltag)
#print info
print "the new tag need be update size is %s"%len(newtag)
#sort the recalltag
recalltag_count= sorted(recalltag_count.items(), key=lambda d:d[1], reverse = False)
#
if len(newtag)>0:
        for idx,tag_to_change in enumerate(newtag.keys()):
                tag_be_change = recalltag_count[idx][0]
                recalltag2code[tag_to_change] = recalltag2code[tag_be_change]
#save
pickle.dump(recalltag2code,open("new_recalltag2code.pkl","wb"))