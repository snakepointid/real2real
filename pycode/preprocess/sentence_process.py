import numpy as np
import sys
import re
def sentenceSeg(sentence):
        sentence = sentence.lower()
        ret,eng,numb = [],'',''
        for char in sentence:
                if re.match("[a-z']",char):
                        eng+=char
                elif re.match("[0-9.]",char):
                        numb+=char
                else:
                        if len(eng):
                                ret+=["#ENG#"];eng=""
                        if len(numb):
                                ret+=["#NUMB#"];numb=""
                        ret+=[char]
        if len(eng):
                ret+=["#ENG#"];eng=""
        if len(numb):
                ret+=["#NUMB#"];numb=""
        return ret
