import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
import random
import sys
import resnet
from skimage.segmentation import mark_boundaries
from torchvision import models, transforms
import pandas as pd
import numpy as np
from sklearn import svm 
import re
from copy import deepcopy
from datetime import datetime
import pickle
import glob
from sklearn.utils import shuffle
import os
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from sklearn.metrics import roc_auc_score
import sys, os, time
import itertools as it
import torch.nn as nn
import argparse
import matplotlib.patches as patches
from collections import deque
import torch.nn as nn
sigmoid = nn.Sigmoid()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def infer(X):
    '''
    this function is to output model's prediction of different layouts
    X: the specific layout
    '''
    model=torch.load('./trained_models/save_model/resnet_120epoch')#the address of your saved model
    model.eval()
    X = X.astype(np.uint8) # necessary!
    toT = transforms.ToTensor()
    X = toT(X).to(torch.float32)
    X = torch.reshape(X, (1, 1, 600, 600))
    logits = model(X.to(device))
    probs = round (sigmoid(logits).detach().cpu().numpy()[0][0], 3)
    final_logits = round (logits.detach().cpu().numpy()[0][0], 2) 
    
    return final_logits

def seg(X): 
    '''
    This function aims to find every entire shape in a layout
    '''
    def it_bfs(x, y, _directions=((-1, 0), (0, -1), (1, 0), (0, 1))):
        queue = deque([(x, y)])
        while queue:
            i, j = queue.popleft()
            if i >= 0 and j >= 0 and i < h and j < w and X[i, j] != 0:
                board[i, j] = X[i, j]
                X[i, j] = 0
            else:
                continue
            for dr, dc in _directions:
                nr, nc = i + dr, j + dc
                queue.append((nr, nc))

    h, w = X.shape
    Xcopy = X.copy()
    rtn = []
    for i in range(w):
        for j in range(h):
            if X[i, j] != 0:
                board = np.zeros((h, w))
                it_bfs (i, j)
                rtn.append(board.copy())    
    return rtn

def permutation(arr2,M):
    '''
    This function aims to find the part that has the greatest contribution
    of layout's attributes. 
    arr2: list that contains the divided shapes.
    M: the original layout
    '''
    p_t1=[]
    for i,X in enumerate(arr2):
        p = infer(X)
        p_t1.append(p)
    p_t1 = np.array(p_t1)
    indexl = np.argsort(-p_t1)
    val_num = []
    area_per = []
    permu_per=[]
    for num in range(1, 5):
        val = []
        each_list = []
        supp = []
        for each in list(it.permutations(indexl,num)):
            sup = np.zeros(360000)
            sup = sup.reshape(600,600)
            each1 = list(each)
            each_list.append(each)
            for i1 in each1:
                sup = sup+arr2[i1]
            supp.append(sup)
            p_e = (infer(sup)-infer(M-sup))
            val.append(p_e)
        index_val = val.index(max(val))
        arr_w = supp[index_val]
        val_num.append(arr_w)
        ara1 = max(val)/(num*num)
        permu_per.append(ara1)
    index_per4 = permu_per.index(max(permu_per))
    arr_re = val_num[index_per4]
    print('value of each permutation',permu_per)
    print('max val is:',max(permu_per))
    print('the number of chose parts is:',index_per4+1)
    arr_re = val_num[index_per4]
    
    return arr_re

def complete(stepc, listt, arr_t, shape_dis):
    '''
    This function aims to modify the chosen part in case it seems incongruous
    listt: list that contains the divided shapes.
    arr_t: the chosen part
    shape_dis: the parameter thats describes the location of shape
    '''
    if shape_dis<0:
        arr_t1 = arr_t.transpose()
    else:
        arr_t1 = arr_t
    for sections in listt: 
        if shape_dis<0:
            sections = sections.transpose()
        else:
            sections = sections
        sections1 = sections.copy()
        shapes = seg(sections)
        lengths = []
        x_leftl = []
        x_rightl = []
        for i1 in shapes:
            x_left = 0
            x_right = 0
            for xl in range(i1.shape[1]):
                if i1[:,xl].sum()>0:
                    x_left = xl
                    x_leftl.append(x_left)
                    break
            for xr in range(i1.shape[1]):
                if i1[:,i1.shape[1]-xr-1].sum()>0:
                    x_right = i1.shape[1]-xr-1
                    x_rightl.append(x_right)
                    break
            length = x_right-x_left
            lengths.append(length)
        if len(lengths)>0:
            if min(lengths)<stepc*0.24:
                print('min', min(lengths), 'step', stepc)
                index_min = lengths.index(min(lengths))
                if sections1[:,x_leftl[index_min]-2].sum()==0:
                    print('Fill it on the right')
                    sup = np.zeros(360000)
                    sup1 = sup.reshape(600,600)
                    sup1[:,x_leftl[index_min]:(x_leftl[index_min]+min(lengths))] = sections1[:,x_leftl[index_min]:(x_leftl[index_min]+min(lengths))]
                    arr_t1 = arr_t+sup1
                if sections[:,x_rightl[index_min]+1].sum() == 0:#0
                    print('Fill it on the left')
                    sup = np.zeros(360000)
                    sup1 = sup.reshape(600,600)
                    sup1[:,(x_rightl[index_min]-min(lengths)):x_rightl[index_min]+1]=sections1[:,(x_rightl[index_min]-min(lengths)):x_rightl[index_min]+1]
                    arr_t1 = arr_t+sup1
    
    if shape_dis<0:
        arr_t1=arr_t1.transpose()
    else:
        arr_t1=arr_t1
    return arr_t1

def segmentation4(arr):
    '''
    This function aims to divide the chose shape 
    arr_t: the chosen shape
    '''
    len1,wid = arr.shape[0],arr.shape[1]
    #row
    for row in range(len1):
        va = arr[row,:].sum()
        if va>0:
            r_u = row
            break
    for row11 in range(len1):
        va2 = arr[len1-row11-1,:].sum()
        if va2>0:
            r_d = len1-row11-1
            break
    r_m = round((r_u+r_d)/2)
    disr = r_d-r_u
    #coloum
    for n in range(wid):
        va3 = arr[:,n].sum()
        if va3>0:
            c_l = n
            break
    for n1 in range(wid):
        va4 = arr[:,wid-n1-1].sum()
        if va4>0:
            c_r = wid-n1-1
            break
    disc = c_r-c_l
    shape_dis = disc-disr
    if disc >= disr:
        stepc = round((c_r-c_l)/4)
        c_ll = c_l
        r_uu = r_u
        r_dd = r_d
    else:
        stepc = round((r_d-r_u)/4)
        c_ll = r_u
        r_uu = c_l
        r_dd = c_r
    parts = []
    for c in range(4):
        #print(c)
        sup = np.zeros(360000)
        sup = sup.reshape(600,600)
        c_l1 = c_ll+c*stepc
        sup[r_uu:r_dd+1,c_l1:c_l1+stepc] = arr[r_uu:r_dd+1,c_l1:c_l1+stepc]
        parts.append(sup)
    return parts, stepc, shape_dis

def apple(X): 
    '''
    This function is the interpreter we created 
    X: the layout you want to examine
    '''
    Xc = X.copy()
    rtn = seg(X)
    p_l1 = []
    p_l2 = []
    p_l3 = []
    rtn2 = []
    rtn3 = []
    print('one shape:')
    for j, Xi in enumerate(rtn):
        p1 = infer(Xi)-infer(Xc-Xi)
        p_l1.append(p1)
    index1 = p_l1.index(max(p_l1))
    max1 = max(p_l1)
    print ('two shapes:')
    for m in range(len(rtn)):
        for n in range(m+1, len(rtn)):
            p2 = infer(rtn[m] + rtn[n])-infer(Xc-rtn[m]-rtn[n])
            p2 = p2/4
            rtn2.append(rtn[m] + rtn[n])
            p_l2.append(p2)
    index2 = p_l2.index(max(p_l2))
    max2 = max(p_l2)
    print ('three shapes:')
    for m in range(len(rtn)):
        for n in range(m+1, len(rtn)):
            for k in range(n+1, len(rtn)):
                p3 = infer(rtn[m] + rtn[n] + rtn[k])-infer(Xc-rtn[m]-rtn[n]-rtn[k])
                p3 = p3/9
                rtn3.append(rtn[m] + rtn[n] + rtn[k])
                p_l3.append(p3)
    index3 = p_l3.index(max(p_l3))
    max3 = max(p_l3)
    print('max3:',max3,'max2:',max2,'max1:',max1)
    max_l = [max1,max2,max3]
    rtn_Xi = [rtn[index1],rtn2[index2],rtn3[index3]]
    index_s1 = max_l.index(max(max_l))
    print('the number of chose shape ',index_s1+1)
    Xi = rtn_Xi[index_s1]
    print('area',Xi.sum())
    if Xi.sum()>1750000:# in case the shape is too small
        list1,step,judge = segmentation4(Xi)
        Xii1 = permutation(list1,Xc)
        Xii = Xii1
        #Xii = complete(step,list1,Xii1,judge)
    else:
        Xii = Xi
    
    return Xii
