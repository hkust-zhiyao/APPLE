import numpy as np
import sys
from functions import Logger
from apple_helper import apple
from functions import metrics
from functions import plot_twoMap
import random
import torch

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sys.stdout = Logger(sys.stdout)
    random.seed(0)
    for bench in range(3,4):# using datasets from iccad2012 contest
        print('Benchmark:',bench)
        X = np.load('saveData/Xtest' + str(bench) + '.npy')
        y = np.load('saveData/Ytest' + str(bench) + '.npy')
        print('X.shape',X.shape)
        sample = y >0 #choose positive samples for interpretation
        X_positive = X[sample]
        y_positive = y[sample]
        print('number of positive samples:', X_positive.shape[0])
        for t in range(len(y_positive)):
            Xt = X_positive[t]
            Xtc = Xt.copy()
            #call apple
            apple_out = apple(Xt) 
            #you can judge model's performance by calling below function
            metrics(Xtc, apple_out)
            #you can visiualize the result by calling below function
            name = 'pictures_new/apple/'+str(bench)+'/'+str(t)+'.png'
            plot_twoMap(apple_out, Xtc, name)
