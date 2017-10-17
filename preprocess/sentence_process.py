#coding=utf-8
import numpy as np
import sys
import re
def quick_sentence_segment(sentence,case_sense=False,keep_eng=False):
        if not case_sense:
                sentence = sentence.lower()
        segmented,eng,numb = [],'',''
        for char in sentence:
                if re.match("[a-z']",char):
                        eng+=char
                elif re.match("[0-9.]",char):
                        numb+=char
                else:
                        if len(eng):
                                if keep_eng:
                                        segmented+=[eng];eng=""
                                else:
                                        segmented+=["#ENG#"];eng=""
                        if len(numb):
                                segmented+=["#NUMB#"];numb=""
                        if char!=' ':
                                segmented+=[char]
        if len(eng):
                if keep_eng:
                        segmented+=[eng];eng=""
                else:
                        segmented+=["#ENG#"];eng=""
        if len(numb):
                segmented+=["#NUMB#"];numb=""
        return segmented

def full_sentence_segment(sentence,case_sense=False,keep_eng=True):
        if not case_sense:
                sentence = sentence.lower()
        segmented,eng,numb,punced = [],'','',True
        for char in sentence:
                if re.match("[a-z']",char):
                        eng+=char;punced=False
                elif re.match("[\u4e00-\u9fa5]",char):
                        segmented+=[char];punced=False
                elif re.match("[0-9.]",char):
                        numb+=char
                else:
                        if len(eng):
                                if keep_eng:
                                        segmented+=[eng];eng=""
                                else:
                                        segmented+=["#ENG#"];eng=""
                        if len(numb):
                                segmented+=["#NUMB#"];numb=""

                        if char in '.,?;:<>[]{}()!。，？；：《》【】（）！'and not punced:
                            segmented+=[char];punced=True
        if len(eng):
                if keep_eng:
                        segmented+=[eng];eng=""
                else:
                        segmented+=["#ENG#"];eng=""
        if len(numb):
                                segmented+=["#NUMB#"];numb=""
        return segmented
