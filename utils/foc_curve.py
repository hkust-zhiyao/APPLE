import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
#from plots import time_scatter
from scipy import linalg
from scipy import sparse
import time
import argparse
import sys
import scipy

def plots(x, y, name, c, m='o', s=13, ls='-', zo=0):
    r = str(round(scipy.stats.pearsonr(x, y)[0], 2))
    k, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    k = str(round(k, 1))

    plt.plot(x, y, c=c, ls=ls, linewidth=3.5,
             marker=m, markersize=s, label=name + r' $(k=' + k + ', R=' + r+ ')$' , zorder=zo)
             #marker=m, markersize=s, label=name + r' $(R=' + r + ', k=' + k+ ')$' , zorder=zo)
    plt.scatter(x, y, c=c, s=[s*10]*(len(x) - 1) + [s *10 * 3], marker=m, zorder=zo)
    return r


def signal():
    from matplotlib import rcParams

    rcParams.update({'figure.autolayout': True})
    plt.figure(figsize=(6.7, 5.6))

    #x      = [0.94,   0.96,   0.98,   1.00]
    x      = [0.942, 0.966, 0.987, 0.996]
    sin_l  = [0.0658, 0.0665, 0.0675, 0.0721]
    sout_l = [0.2790, 0.2734, 0.2632, 0.2402]
    win_l  = [0.1071, 0.1251, 0.1270, 0.1601]

    sin_a  = [0.1222, 0.1588, 0.2498, 0.2800]
    sout_a = [0.4041, 0.3417, 0.3282, 0.1803]
    win_a  = [0.2831, 0.3598, 0.6284, 0.6612]

    ############################################################################
    ax = plt.gca()

    #ss = [12] * (len(x)-1) + [24]
    #print ('ss', ss)

    plots(x, sin_l,  name= r'$FC_{in}$' + '_LIME   ' + r'$\:$' + r'$\uparrow$', c='orange', m='o', s=12, ls=':', zo=0)
    plots(x, sin_a,  name= r'$FC_{in}$' + '_APPLE' + r'$\:\:$' + r'$\uparrow$', c="xkcd:" + 'turquoise', m='o', s=12, zo=2)

    plots(x, win_l, name= r'$FC_{inE}$' + '_LIME ' + r'$\:\:$' + r'$\uparrow$', c='orange', m='*', s=16, ls=':', zo=1)
    plots(x, win_a, name= r'$FC_{inE}$' + '_APPLE' + r'$\uparrow$', c="xkcd:" + 'turquoise', m='*', s=16, zo=3)

    plt.xlabel('Model Accuracy (ROC AUC)', fontsize=22)
    plt.ylabel('FC Value', fontsize=22)

    #plt.xticks([0, 0.6, 1.2, 1.8], ['0', '.6', '1.2', '1.8'], fontsize=20 )
    plt.xticks([.94, .96, .98, 1.0], fontsize=20 )
    plt.xlim([0.937, 1.002])


    plt.yticks(fontsize=20)
    ax.set_yticks([0.1, 0.4, 0.7])
    plt.ylim([0, 1.18])

    #plt.legend(fontsize=20, ncol=2, columnspacing=1.0, 
    legend = plt.legend(fontsize=19, ncol=1, columnspacing=0.5, 
               labelspacing=0.4, borderaxespad=0.2, loc='upper right')

    legend.get_frame().set_linewidth(5)

    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(2.5)

    plt.savefig('FOC_curve', dpi=200)
    print ('save:', 'FOC_curve')
    plt.close('all')
    plt.clf()


signal()





