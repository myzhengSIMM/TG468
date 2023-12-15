import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import os
import random
import numpy as np
import matplotlib.pyplot as plt

seed = 10
os.environ['PYTHONHASHSEED'] = str(seed)    
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
np.random.seed(seed) 
random.seed(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  
torch.backends.cudnn.enabled = True 

test_set_name = 'Hellmann'   #  or Liu  
time = 'PFS_MONTHS'          #  OS_MONTHS
time_status = 'PFS_STATUS'   #  OS_STATUS

train_set = pd.read_csv('../data/train_set.csv')
train_set = shuffle(train_set)
test_set = pd.read_csv('../data/test_set.csv')

MUTsentence = pd.concat([train_set,test_set],axis=0)

lines = list(MUTsentence['mutsentence'])
label = list(MUTsentence['label'])
label = np.array(label)
Lines = [line.split() for line in lines]
selected_gene  = pd.read_csv('../data/candidate gene set.csv')
selected_gene = selected_gene[selected_gene['p_value']<=0.05]
selected_gene = list(selected_gene['gene_name'])
Sentence = []
for line in Lines:
    new_sentence = []
    for word in selected_gene:
        if word in line[0]:
            new_sentence.append(1)
        else:
            new_sentence.append(0)
    Sentence.append(new_sentence)
Sentence = np.array(Sentence)
Sentence.shape

TRAIN_SIZE = train_set.shape[0]
X_train, y_train = Sentence[:TRAIN_SIZE], label[:TRAIN_SIZE]
X_test, y_test = Sentence[TRAIN_SIZE:],label[TRAIN_SIZE:]
X_train = pd.DataFrame(data=X_train, columns=selected_gene)
y_train = pd.DataFrame(data=y_train, columns=['target'])
X_test = pd.DataFrame(data=X_test, columns=selected_gene)
y_test = pd.DataFrame(data=y_test, columns=['target'])
"SVM"
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

"kNN"
# from sklearn.neighbors import KNeighborsClassifier
# model= KNeighborsClassifier()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

"RandomForest"
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# model.fit(X_train, y_train.values.ravel())
# y_pred = model.predict(X_test)

"Bayesian classifier"
# from sklearn.naive_bayes import ComplementNB  
# model = ComplementNB()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

time_info = test_set

test_PFS = list(time_info[time])
test_state = list(time_info[time_status])
realLabel = list(time_info['label'])
ID = list(time_info['SAMPLE_ID'])

test_pred = test_set
test_pred['time'] = test_PFS
test_pred['state'] = test_state
test_pred['realLabel'] = realLabel
test_pred['ID'] = ID
test_pred.replace('0:Censor','0',inplace = True)
test_pred.replace('1:Event','1',inplace = True)

test_pred['response_pred'] = y_pred  

test_pred['Gender'] = list(test_set['Gender']) 
if test_set_name == 'Hellmann':
    AGE = []
    for a in list(test_set['Age']):
        if a>=65:
            AGE.append(1)
        elif a<65:
            AGE.append(0)
    test_pred['AGE'] = AGE

TMB = []
for a in list(test_set['nonsynonymous_somatic_mutations_counts']):
    if a>=380:
        TMB.append(1)
    elif a<380:
        TMB.append(0)
test_pred['TMB'] = TMB

test_pred.to_csv('../Results/pred-response.csv',index = False)

from sklearn.metrics import confusion_matrix

TN,FP,FN,TP = confusion_matrix(y_test,y_pred).ravel()
sensitivuty = TP/(TP+FN)  
specificity = TN/(TN+FP)  
npv,ppv = round(TN/(FN+TN),2),round(TP/(TP+FP),2) 
print('sensitivuty,specificity = ',sensitivuty,specificity)
print('Accuracy = ', (TP + TN)/(TN+FP+FN+TP))      
print('ppv,npv = ',ppv,npv)    
print('portion = ' , y_pred.sum(), round(y_pred.sum()/y_pred.shape[0],4))
