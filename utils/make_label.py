import numpy as np
import os
from tqdm import tqdm
import cv2
import torch
from PIL  import Image
import random



out_save=[]
png_dir='hs_model/attack.p20.test-H_all/'
#set label
hot=np.array([1])
nhot=np.array([0])
count=0

for png in tqdm(os.listdir(png_dir), desc='Making label'):
    png_path = os.path.join(png_dir, png)
    count+=1
    if png[-5]=='1':
        img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(600,600))
        print('img',img)
        img=(img*255).astype('uint8')
        img= torch.from_numpy(img).view(1, 1, *img.shape).float() #cuda from_numpy
        out2=img.reshape(1,-1).numpy()

        out2=out2.flatten()
        out3=np.hstack((out2,nhot))
        out_save.append(out3)
np.save('test_adver_1.npy',out_save)


#merge datasets
# random.seed(100)
# x1=np.load('test_back1_h_v.npy')
# print('x1',x1.shape)
# x2=np.load('test_clean_n.npy')
# print('x2',x2.shape)
# x5=np.vstack((x1,x2))
# print('shape3',x3.shape)
# np.random.shuffle(x3)
# np.save('test_back1_small_v.npy',x3)


