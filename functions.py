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
import torch.nn as nn
sigmoid = nn.Sigmoid()
from dct_30 import dct_30
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




# log recorder
class Logger(object):

    def __init__(self, stream=sys.stdout):
        output_dir = "./log_print/2/"  # the address of saving log
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        log_name = log_name_time + ".txt"
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


#visualize the result
def plot_twoMap(x, xback, name = 'no'):
    
    x = x > 0
    xback = xback > 0
    x_all = xback + x * 10
    x_all=x_all#.transpose()

    cmap = colors.ListedColormap(['black', 'white', 'green'])
    bounds=[0, 0.1, 10, 100] #district/interval
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.figure()
    img = plt.imshow(x_all, cmap=cmap, norm=norm)
    ax = plt.gca()
    # rect=patches.Rectangle((220,220),160,160,edgecolor='orange',linewidth=6,facecolor='none',linestyle='dotted')
    # ax.add_patch(rect)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    plt.savefig(name, dpi = 200, bbox_inches='tight')
    plt.close()
    plt.clf()
    
def sub_plot(target_num,ROW_NUM,COL_NUM,png_a,png_b,png_c,png_n):            
    target_count = 0                
    for plot_num in range(0, target_num, ROW_NUM * COL_NUM):
        figure, axis = plt.subplots(nrows=ROW_NUM, ncols=COL_NUM, sharex='all', sharey='all',figsize=(6,6))   
        sub_figure_index = []
        for i in range(axis.shape[0]):
            for j in range(axis.shape[1]):
                sub_figure_index.append(axis[i, j])
        for index in range(ROW_NUM * COL_NUM):
            if target_count == target_num:
                break
            sub_figure_index[index].imshow(png_a[index],png_b[index],png_c[index])
            sub_figure_index[index].set_xlabel(png_n[index])
            target_count += 1
        plt.tight_layout()
        plt.savefig(name.format(plot_num / (ROW_NUM * COL_NUM)), dpi=200)
        print('successful!')
        
def infer(X): 
    X = X.astype(np.uint8) # necessary!
    toT = transforms.ToTensor()
    X = toT(X).to(torch.float32)
    X = torch.reshape(X, (1, 1, 600, 600))
    logits = resnet(X.to(device))
    probs = round (sigmoid(logits).detach().cpu().numpy()[0][0], 3)
    final_logits = round (logits.detach().cpu().numpy()[0][0], 2) #output of our model
    return final_logits

def metrics(Xc,Xii):
    #metric
    Xc[Xc>0]=255
    Xii[Xii>0]=255
    focus_area_all=Xii.sum()
    focus_area_inter=Xii[220:380,220:380].sum()
    wire_area_all=Xii.sum()
    wire_area_inter=Xii[220:380,220:380].sum()
    Xc_all=Xc.sum()
    Xc_inter=Xc[220:380,220:380].sum()
    
    
    con_np=np.ones((600,600),dtype=float)
    con_np=con_np*255
    cons=con_np[220:380,220:380].sum()
    
    print('')
    print('apple************************************')  
    print('inter focus area',focus_area_inter)
    print('all focus area',focus_area_all)
    print('all focus area wire',wire_area_all)
    print('inter focus area wire',wire_area_inter)
    print('shape in',round(focus_area_inter/cons,4))
    print('shape out',round(((focus_area_all-focus_area_inter)/cons),4))
    print('wire in',round(wire_area_inter/Xc_inter,4))
    print('apple************************************')
    print('')
    
    

    

    
