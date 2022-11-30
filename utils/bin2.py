
import numpy as np
from plot2_helper import plot_acc
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def flipTau(n):
    return [i + (1-i)/2 for i in n]

#total = [18, -24]
#part = [18, -38]
#other = [-2, -23]

        # p,     n,    an
total = [5.21, -23.26, -15.34]
part =  [5.28, -34.22, -45.15]
other = [-31.7, -21.17, -1.55]

print ('test2')
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)
plot_acc (total, part, other, name='bin2')

