from __future__ import division
from __future__ import print_function
from sklearn import metrics
import random
import time
import sys
import os
sys.path.append('..')


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np



from utils.utils import *
from models.gcn import GCN

from config.config import CONFIG
from sklearn.metrics import roc_curve
import pandas as pd
cfg = CONFIG()



# Set random seed
seed = cfg.seed           

os.environ['PYTHONHASHSEED'] = str(seed)    
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  
random.seed(seed)  
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False



datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr','gs'] 
dataset = 'gs' 

if dataset not in datasets:
	sys.exit("wrong dataset name")
cfg.dataset = dataset



# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
    cfg.dataset) 
features = sp.identity(features.shape[0])  



# Some preprocessing
features = preprocess_features(features)   
if cfg.model == 'gcn':
    support = [preprocess_adj(adj)]   
    num_supports = 1
    model_func = GCN  
elif cfg.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, cfg.max_degree)
    num_supports = 1 + cfg.max_degree
    model_func = GCN
elif cfg.model == 'dense': 
    support = [preprocess_adj(adj)]  
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(cfg.model))


t_features = torch.from_numpy(features)
t_y_train = torch.from_numpy(y_train)
t_y_val = torch.from_numpy(y_val)
t_y_test = torch.from_numpy(y_test)
t_train_mask = torch.from_numpy(train_mask.astype(np.float32))
tm_train_mask = torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0)#.repeat(1, y_train.shape[1] 

t_support = []

for i in range(len(support)):
    t_support.append(torch.Tensor(support[i]))

# if torch.cuda.is_available():
#     model_func = model_func.cuda()
#     t_features = t_features.cuda()
#     t_y_train = t_y_train.cuda()
#     t_y_val = t_y_val.cuda()
#     t_y_test = t_y_test.cuda()
#     t_train_mask = t_train_mask.cuda()
#     tm_train_mask = tm_train_mask.cuda()
#     for i in range(len(support)):
#         t_support = [t.cuda() for t in t_support if True]
        
model = model_func(input_dim=features.shape[0], support=t_support, num_classes=1) 
criterion  = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)


def evaluate(features, labels, mask):  
    t_test = time.time()
    model.eval()
    with torch.no_grad():
        logits = model(features)
        t_mask = torch.from_numpy(np.array(mask*1., dtype=np.float32))  
        tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0)
        logits = logits * tm_mask   
        loss = criterion(logits.view(-1,1), labels.view(-1,1).to(torch.float32)).sum()
        pred = logits.view(-1,1) 
    model.train()    
    return loss.numpy(), pred.numpy(), labels.numpy(), (time.time() - t_test)


from sklearn.metrics import roc_auc_score
train_losses = []
TRAIN_AUC = []
val_losses = []
VAL_AUC = []
TEST_AUC = []
SMOOTH = 0.0

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, eps=1e-4, verbose=True)

for epoch in range(cfg.epochs):
    t = time.time()
    logits = model(t_features)
    logits = logits * tm_train_mask      
    t_y_train_s  = t_y_train
    loss = criterion(logits.view(-1,1),t_y_train_s.to(torch.float32)).mean()
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    train_losses.append(loss.detach().numpy())
    t_y_val_S = t_y_val.view(-1,1)*(1-SMOOTH) + (1-t_y_val.view(-1,1))*SMOOTH
    val_loss, pred, labels, duration = evaluate(t_features, t_y_val_S, val_mask)
    _,pred_train, labels_train, _ = evaluate(t_features, t_y_train, t_train_mask)
    valid_loss, pred_v, labels_v, valid_duration = evaluate(t_features, t_y_val, val_mask)

    valid_pred = []
    valid_labels = []
    val_mask.shape
    for i in range(len(val_mask)):
        if val_mask[i]:
            valid_pred.append(pred_v[i])
            valid_labels.append(labels_v[i])
    val_auc = roc_auc_score(valid_labels, valid_pred)
    VAL_AUC.append(val_auc)
    val_losses.append(val_loss)
    
    train_labels = []
    train_pred = []
    for i in range(len(t_train_mask)):   
        if t_train_mask[i]:
            train_pred.append(pred_train[i])
            train_labels.append(labels_train[i])
    train_auc = roc_auc_score(train_labels, train_pred)
    TRAIN_AUC.append(train_auc)
    

    print_log("Epoch: {:.0f}, train_loss= {:.5f}, val_loss= {:.5f}, time= {:.5f}"\
                .format(epoch + 1, loss, val_loss, time.time() - t))
    

    if cfg.test_set_name == 'Liu': 
        if epoch > 200 and epoch > cfg.early_stopping and (val_losses[-1] > np.mean(val_losses[-(cfg.early_stopping+1):-1])):
            break
    else:
        if epoch > cfg.early_stopping and (val_losses[-1] > np.mean(val_losses[-(cfg.early_stopping+1):-1])):
            break

test_loss, pred, labels, test_duration = evaluate(t_features, t_y_test, test_mask)
print_log("Test set results: \n\t loss= {:.5f},time= {:.5f}".format(test_loss, test_duration))

valid_loss, pred_v, labels_v, valid_duration = evaluate(t_features, t_y_val, val_mask)



test_pred = []
test_labels = []
for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(labels[i])



print('auc_test ',roc_auc_score(test_labels, test_pred)) 
print('auc_valid',roc_auc_score(valid_labels, valid_pred))

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
    save_pred.to_csv('../results/ICI-pred_' + data_catalog + '.csv',index = False)


get_roc_coordinatesANDpred(train_labels, train_pred,'train')  
get_roc_coordinatesANDpred(valid_labels, valid_pred,'valid')  
get_roc_coordinatesANDpred(test_labels, test_pred,'test')   



