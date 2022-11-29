import pandas as pd
import numpy as np
from sklearn import svm
import sys
import re
from copy import deepcopy
from datetime import datetime
import pickle
import glob
from sklearn.utils import shuffle
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
torch.manual_seed(0)
random.seed(0)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-lr", "--lr", help = "provide learning rate", type=float, default=0.01)
parser.add_argument("-split", "--split", help = "provide training/testing split", type=str, default=None)
parser.add_argument("-use_test", "--use_test", help = "whether using test data of training benchmark for training", type=int, default=1)


args = parser.parse_args()
print ('args.use_test, whether use test data in training benchmark for training', args.use_test)

retrain_and_save_models = True # if True: train locally and save models at model_path. if False: pre-trained models from model_path will be used

XX_train, yy_train, XX_test, yy_test = [], [], [], []

device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# X_train=np.load('/home/tzhangbw/litho/full_litho/benchmark-litho/train_back1_small2.npy')[:,:-1]
# y_train=np.load('/home/tzhangbw/litho/full_litho/benchmark-litho/train_back1_small2.npy')[:,-1]
# X_test=np.load('/home/tzhangbw/litho/full_litho/benchmark-litho/test_clean_small2.npy')[:,:-1]
# y_test=np.load('/home/tzhangbw/litho/full_litho/benchmark-litho/test_clean_small2.npy')[:,-1]
# X_train=X_train.reshape(-1,600,600)
# X_test=X_test.reshape(-1,600,600)
# #train
# X_train=X_train[0:10000,:]
# # X_train2=X_train[10000:20001,:]
# y_train=y_train[0:10000]
# # y_train2=y_train[20001:]
# # X1=[X_train1,X_train2]
# # y1=[y_train1,y_train2]
# # #test
# X_test=X_test[0:10000,:]
# # X_test2=X_test[10000:20001,:]
# y_test=y_test[0:10000]
# # y_test2=y_test[20001:]
# # X2=[X_test1,X_test2]
# # y2=[y_test1,y_test2]




# for original dataset:
sel = 2000
for i in range(1, 3):#(1,6)
    #X=np.load('/home/tzhangbw/litho/full_litho/benchmark-litho/train_clean.npy')[:,:-1]
    X = np.load('saveData/Xtest' + str(i) + '.npy')
    y = np.load('saveData/Ytest' + str(i) + '.npy')
    #y=np.load('/home/tzhangbw/litho/full_litho/benchmark-litho/train_clean.npy')[:,-1]
    # X = np.load('saveData/Xtrain' + str(i) + '.npy')
    # y = np.load('saveData/Ytrain' + str(i) + '.npy')
    # X=X1[i-1]
    # y=y1[i-1]
    L = X.shape[0]
    print ('X.shape', X.shape)
    sample = random.sample(list(range(L)), min(sel, L))
    X = X[sample]
    y = y[sample]
    XX_train.append(X)
    yy_train.append(y)


sep_list = [0]
test_benches = []
for i in range(1, 6):#(1,4)
    # X=X2[i-1]
    # y=y2[i-1]
    #X=np.load('/home/tzhangbw/litho/full_litho/benchmark-litho/test_clean.npy')[:,:-1]
    X = np.load('saveData/Xtest' + str(i) + '.npy')
    y = np.load('saveData/Ytest' + str(i) + '.npy')
    #y=np.load('/home/tzhangbw/litho/full_litho/benchmark-litho/test_clean.npy')[:,-1]
    
    print ('X.shape', X.shape)
    #mask_negative = (y < 1).astype(int)
    #L = len(mask_negative)
    #sample = random.sample(list(range(L)), min(sel, L))

    #mask_negative[sample] += 1
    #print ('mask_negative', mask_negative.max())
    #mask_negative -= 1
    #mask_negative = mask_negative > 0

    #mask_positive = y > 0
    #print ('mask_positive, mask_negative', mask_positive.sum(), mask_negative.sum())
    #sample_all = mask_negative + mask_positive

    L = X.shape[0]
    print ('X.shape', X.shape)
    sample_all = random.sample(list(range(L)), min(sel, L))

    X = X[sample_all]
    y = y[sample_all]
    XX_test.append(X)
    yy_test.append(y)
    sep_list.append(sep_list[-1] + len(y))
    test_benches.append(i)

X_train = np.vstack(XX_train)
y_train = np.hstack(yy_train)
X_test = np.vstack(XX_test)
y_test = np.hstack(yy_test)





print ('X_train, y_train', X_train.shape, y_train.shape, y_train.sum())
print ('X_test, y_test', X_test.shape, y_test.shape, y_test.sum())

class LithoDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def setupTrans(self):
        self.toT =transforms.ToTensor()
        self.hori = transforms.RandomHorizontalFlip()
        self.vert = transforms.RandomVerticalFlip()
        #self.resize = transforms.Resize((224, 224))
        self.trans = transforms.Compose([self.toT, self.hori, self.vert])
        #self.trans = transforms.Compose([self.toT, self.hori, self.vert, self.resize])

    def __getitem__(self, idx):
        #print ('X.sum 1', idx, self.X[idx].sum(), self.X[idx].max(), type(self.X[idx]), self.X[idx].dtype)
        self.setupTrans()
        X_conv = self.trans(self.X[idx]).to(torch.float32)
        #print ('X.sum 2', idx, X_conv.sum(), X_conv.max(), type(X_conv), X_conv.dtype)
        #exit()
        # it convert range from 255 (uint8) to 1 (float 32) !!!
        return self.trans(self.X[idx]).to(torch.float32), self.y[idx]


dataset = LithoDataset(X=X_train,
                        y=y_train)
test_dataset = LithoDataset(X=X_test,
                            y=y_test)

loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=16,
                                     shuffle=True,
                                     num_workers=8)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                     batch_size=6,
                                     shuffle=False,
                                     num_workers=8)



from resnet import getModelNet600
resnet = getModelNet600() # default one output
#resnet=RandomForestClassifier(n_estimators=300)

#loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.BCELoss()

print ('learning rate:', args.lr)
optimizer = torch.optim.Adam(resnet.parameters(),lr =0.0001,weight_decay = 1e-8)



resnet.to(device)

import torch.nn as nn
sigmoid = nn.Sigmoid()

#print ('test loss_fn 4', loss_fn( torch.FloatTensor([0, 0, 0, 0]), torch.FloatTensor([0.5, 0.5, 0.5, 0.5]) ))
#print ('test minus', 1 - torch.FloatTensor([0, 0, 0.5, 1]) )

num_epoches =9
for epoch in range(num_epoches):
    resnet.train()
    train_loss = []
    val_loss = []

    TP, FP, TN, FN, = 0, 0, 0, 0
    outs, ys, ps = [], [], []
    for idx, (x, y) in enumerate(loader):
        
        x = x.to(device)
        y = y.to(torch.float32).squeeze().to(device)
        if True:
            out = resnet(x).squeeze()
            out = sigmoid(out)
            loss = loss_fn(out, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        outs.append(out.detach().cpu())
        ys.append(y.detach().cpu())
        train_loss.append(loss.detach().cpu().numpy())

    outs = np.hstack(outs)
    ys = np.hstack(ys)

    auc = round(roc_auc_score(ys, outs), 3)
    ps = outs > np.percentile(outs, 100 - 100 * np.sum(ys) / len(ys))

    mask_p = ps == 1
    mask_n = ps == 0
    TP += ((ps == ys) [mask_p]).sum()
    TN += ((ps == ys) [mask_n]).sum()
    FP += ((ps != ys) [mask_p]).sum()
    FN += ((ps != ys) [mask_n]).sum()

    print ('Train: TP, TN, FP, FN', TP, TN, FP, FN)
    print ('Epoch ' + str(epoch) + ', train:', round(np.mean(train_loss), 3), 'auc: ', auc)
    print('accuracy of hotspots:',round(TP/(TP+FN),3))
    print('accuracy of non_hotspots:',round(TN/(TN+FP),3))
    print ()
    
    ##########################################################
    outs, ys, ps = [], [], []
    
    resnet.eval()
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(torch.float32).squeeze().to(device)
        if True:
            out = resnet(x).squeeze()
            out = sigmoid(out)
           
        loss = loss_fn(out, y.float())
        val_loss.append(loss.detach().cpu().numpy())
        outs.append(out.detach().cpu())
        ys.append(y.detach().cpu())

    outs_all = np.hstack(outs)
    ys_all = np.hstack(ys)

    #for i in range(len(sep_list) - 1):
    TP, FP, TN, FN, = 0, 0, 0, 0
    #st, et = sep_list[i], sep_list[i+1]

    outs = outs_all# [st: et]
    ys = ys_all #[st:et]
    auc = round(roc_auc_score(ys, outs), 3)
    ps = outs > np.percentile(outs, 100 - 100 * np.sum(ys) / len(ys))

    mask_p = ps == 1
    mask_n = ps == 0
    TP += ((ps == ys) [mask_p]).sum()
    TN += ((ps == ys) [mask_n]).sum()
    FP += ((ps != ys) [mask_p]).sum()
    FN += ((ps != ys) [mask_n]).sum()
    #print ('testbench:', test_benches[i])
    print ('Val: TP, TN, FP, FN', TP, TN, FP, FN)
    print ('Epoch ' + str(epoch) + ', val:', round(np.mean(val_loss), 3), 'auc: ', auc)
    print('accuracy of hotspots:',round(TP/(TP+FN),3))
    print('accuracy of non_hotspots:',round(TN/(TN+FP),3))
    print ()
    print ('------------------------------\n')

    # if epoch == 20:
    #     resnet.eval()
    #     torch.save(resnet, 'save_model/resnet_20epoch')

    # if epoch > 30:
resnet.eval()
torch.save(resnet, 'trained_models/save_model/resnet_0.96')
       # break
    #print("finished")
