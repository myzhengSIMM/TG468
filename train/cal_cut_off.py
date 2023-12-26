import numpy as np
import pandas as pd
import torch                                   
import warnings
warnings.filterwarnings("ignore", category=Warning)
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score, roc_curve, confusion_matrix

test_set_name = 'Hellmann'  # Liu
time = 'PFS_MONTHS'   #  OS_MONTHS
status = 'PFS_STATUS' #  OS_STATUS

test_pred = pd.read_csv('../results/ICI-pred_test.csv')
train_pred = pd.read_csv('../results/ICI-pred_train.csv')
valid_pred = pd.read_csv('../results/ICI-pred_valid.csv')
reference_set_pred = pd.concat([train_pred,valid_pred],axis = 0)   
test_labels = []
fpr, tpr, thresholds = roc_curve(reference_set_pred['labels'], reference_set_pred['prob'])
youden = tpr-fpr
cutoff = thresholds[np.argmax(youden)]   

for r in list(test_pred['prob']):
    if r >= cutoff:    
        test_labels.append('R')
    else:
        test_labels.append('NR')
test_pred['response_pred'] = test_labels  

time_info = pd.read_csv('../data/test_set.csv')

Time = list(time_info[time])
Status = list(time_info[status])
realLabel = list(time_info['label'])
ID = list(time_info['SAMPLE_ID'])
log2TMB = list(np.log2(time_info['nonsynonymous_somatic_mutations_counts'] + 1.0))

test_pred['time'] = Time
test_pred['state'] = Status
test_pred['realLabel'] = realLabel
test_pred['ID'] = ID
test_pred.replace('0:Censor','0',inplace = True)
test_pred.replace('1:Event','1',inplace = True)
test_pred['log2TMB'] = log2TMB
test_pred['log2NAL'] = list(np.log2(time_info['NAL'] + 1.0))
test_pred.to_csv('../results/pred-response.csv',index = False)
TN,FP,FN,TP = confusion_matrix(test_labels,test_pred['realLabel']).ravel()
sensitivuty = TP/(TP+FN) 
specificity = TN/(TN+FP)  
npv,ppv = round(TN/(FN+TN),2),round(TP/(TP+FP),2) 
print('sensitivuty,specificity = ',sensitivuty,specificity)
print('Accuracy = ', (TP + TN)/(TN+FP+FN+TP))     
print('ppv,npv = ',ppv,npv)   


