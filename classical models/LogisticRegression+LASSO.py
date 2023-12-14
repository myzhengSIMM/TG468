
import pandas as pd
from collections import Counter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso


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


TRAIN_SIZE = train_set.shape[0]
X_train, y_train = Sentence[:TRAIN_SIZE], label[:TRAIN_SIZE]
X_test, y_test = Sentence[TRAIN_SIZE:],label[TRAIN_SIZE:]


"LogisticRegression"
moedel = LogisticRegression()
"LASSO"
#moedel = Lasso(alpha = 0.0)


moedel.fit(X_train, y_train)

y_pred_test = moedel.predict(X_test)
y_pred_train = moedel.predict(X_train)


def get_roc_coordinatesANDpred(labels_true,prob_pred,data_catalog):
    fpr, tpr, threshold = roc_curve(labels_true,prob_pred)
    save = pd.DataFrame()
    save['fpr'] = fpr
    save['tpr'] = tpr
    save['threshold'] = threshold
    save.to_csv('../results/ICI-roc_' + data_catalog + '.csv',index = False)
    save_pred = pd.DataFrame()
    save_pred['labels'] =  [labels_true[j][0] for j in range(len(labels_true))]
    save_pred['prob'] = [prob_pred[j][0] for j in range(len(prob_pred))]
    save_pred.to_csv('../results/ICI-roc_' + data_catalog + '.csv',index = False)
    
get_roc_coordinatesANDpred(y_train, y_pred_train,'train')  
get_roc_coordinatesANDpred(y_test, y_pred_test,'test')   


