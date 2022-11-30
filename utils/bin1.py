
import numpy as np
from plot1_helper import plot_acc
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def flipTau(n):
    return [i + (1-i)/2 for i in n]

#pos_sample = [18, 18, -2]
#neg_sample = [-24, -38, -23]

total = [18, -24, 0]
part = [18, -38, 0]
other = [-2, -23, 0]

print ('test1')
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)
plot_acc (total, part, other, name='bin1')

