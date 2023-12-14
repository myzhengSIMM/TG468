
import pandas as pd
from collections import Counter
from tqdm import tqdm
import numpy as np
import random
import os

from sklearn.utils import shuffle
import sys

sys.path.append('../config') 

from config import CONFIG
CONFIG = CONFIG()

seed0 = CONFIG.seed0
os.environ['PYTHONHASHSEED'] = str(seed0)   
random.seed(seed0)
np.random.seed(seed0)
size = CONFIG.size


test_set_name = CONFIG.test_set_name

train_set = pd.read_csv('../data/train_set.csv')
train_set = shuffle(train_set)
TRAIN_SIZE = train_set.shape[0]

test_set = pd.read_csv('../data/'+test_set_name+'.csv')

MUTsentence = pd.concat([train_set,test_set],axis=0)

lines = list(MUTsentence['mutsentence'])
label = list(MUTsentence['label'])

Lines = [line.split() for line in lines]
      
selected_gene  = pd.read_csv('../data/candidate gene set.csv')
selected_gene = selected_gene[selected_gene['p_value']<=0.05]
selected_gene = list(selected_gene['gene_name'])

Sentence = []
for line in Lines:
    new_sentence = []
    for word in selected_gene:
        if word not in line[0]:
            new_sentence.append(word)
        try:
            if len(new_sentence)>= size:   
                    break               
        except:
            pass
    Sentence.append(new_sentence)

f = open("../data/corpus/gs.txt",'w')
for data in Sentence:
    random.shuffle(data)
    sentence = ' '.join(data)
    f.write(sentence) 
    f.write("\n")
f.close()

f_label = open("../data/gs.txt",'w')
j = 0
num = len(label)
for l in label:
    f_label.write(str(j))
    f_label.write('\t')
    if j<TRAIN_SIZE:  
        f_label.write('train')
        f_label.write('\t') 
        f_label.write(str(l))
        f_label.write("\n")
        j = j+1
    else:   
        f_label.write('test')
        f_label.write('\t')          
        f_label.write(str(l))
        f_label.write("\n")
        j = j+1
f_label.close()