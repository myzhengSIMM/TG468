
import pandas as pd
from collections import Counter
from tqdm import tqdm
import numpy as np

from sklearn.utils import shuffle
import torch
import torch.nn as nn
import random
import torch.utils.data as Data
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import time
from sklearn.metrics import roc_curve
import os


seed = 100


os.environ['PYTHONHASHSEED'] = str(seed)   
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed) 
random.seed(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False 
torch.backends.cudnn.enabled = True


    
    
train_set = pd.read_csv('../data/train_set.csv')
train_set = shuffle(train_set)

test_set = pd.read_csv('../data/test_set.csv')

def get_fearures_and_labels(MUTsentence):  
    lines = list(MUTsentence['mutsentence'])
    label = list(MUTsentence['label'])
    label = np.array(label)
    label = torch.FloatTensor(label)
    label = label.view(-1,1)
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
    Sentence = torch.FloatTensor(Sentence)  
    Sentence.shape
    return Sentence,label

net = nn.Sequential(
            nn.Linear(983, 200), 
            nn.ReLU(), 
            nn.Linear(200, 1),
            nn.Sigmoid(),
            )

   
def train(net,data_iter,optimizer,device,
          train_features,train_targets,valid_features,valid_labels,num_epochs = 0):
   
    train_AUC = []
    valid_AUC = []
    train_loss,valid_loss = [],[]
    
    net = net.to(device)
    train_features = train_features.to(device)
    train_targets = train_targets.to(device)
    valid_features = valid_features.to(device)
    valid_labels = valid_labels.to(device)

    loss = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, eps=1e-4, verbose=True)
    for epoch in range(num_epochs):
        train_l_sum, batch_count, start = 0.0, 0, time.time()
        net.train()
        for X,y in data_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l_batch = loss(y_hat,y) 
            optimizer.zero_grad()
            l_batch.backward()
            optimizer.step()
            train_l_sum += l_batch
            batch_count += 1
        scheduler.step(train_l_sum)
        
        with torch.no_grad():
            net.eval()
            y_train_hat = net(train_features)
            l_train = loss(y_train_hat,train_targets)
            train_loss.append(l_train.to('cpu'))
            auc_train = roc_auc_score(train_features.long().to('cpu').view(-1,1),train_targets.view(-1,1).to('cpu'))
            train_AUC.append(auc_train)
            
            y_valid_hat = net(valid_features)
            l_valid = loss(y_valid_hat,valid_labels)
            valid_loss.append(l_valid.to('cpu'))
            auc_valid = roc_auc_score(valid_labels.view(-1,1).long().to('cpu'),y_valid_hat.view(-1,1).to('cpu'))
            valid_AUC.append(auc_valid)
            net.train()
        print('epoch %d, loss train =  %.7f,auc_train = %.7f, time %.1f sec'
                  % (epoch + 1, l_train, auc_train, time.time() - start))
        if epoch > 10 and (valid_loss[-1] > np.mean(valid_loss[-(10+1):-1])):
            break
    return y_train_hat.to('cpu'), y_valid_hat.to('cpu')


train_features0,train_labels0 =  get_fearures_and_labels(train_set)
train_features0.shape

valid_size = round(train_features0.shape[0]*0.8)

train_features = train_features0[:valid_size]
train_labels = train_labels0[:valid_size]

valid_features = train_features0[valid_size:]
valid_labels = train_labels0[valid_size:]

test_features,test_labels = get_fearures_and_labels(test_set)      

dataset = Data.TensorDataset(train_features,train_labels)
data_iter = Data.DataLoader(dataset,batch_size =20, shuffle=True,num_workers=0)
    
num_epochs = 1000
optimizer =  torch.optim.Adam(net.parameters(),lr = 0.00001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_pred,valid_pred = train(net,data_iter,optimizer,device,
                              train_features,train_labels,valid_features,valid_labels,num_epochs)
net.eval()
test_pred = net(test_features)

def get_roc_coordinatesANDpred(labels_true,prob_pred,data_catalog):
    #print(labels_true)
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


get_roc_coordinatesANDpred(train_labels.numpy(), train_pred.numpy(),'train')  
get_roc_coordinatesANDpred(valid_labels.numpy(), valid_pred.numpy(),'valid')  
get_roc_coordinatesANDpred(test_labels.numpy(), test_pred.numpy(),'test')   
    
