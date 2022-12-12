import os, random, shutil
import numpy as np
import imageio
import os
from skimage.transform import resize
import matplotlib.pyplot as plt
import PIL.Image as Image
import os

def copyFile(fileDir,tarDir):
    '''
    this function is to sample certrain number of images for splicing
    fileDir: the address of original datasets
    tarDir: the address of saving new datasets

    '''
    pathDir = os.listdir(fileDir)  
    filenumber = len(pathDir)
    i = 0
    picknumber = 4  
    sample = random.sample(pathDir, picknumber)  
    print(sample)

    for name in sample:
        shutil.copy(fileDir + name, tarDir + name)
    return 

k = 0
y = []
x = []
num = []
lxy = 0
dest_dir = "/home/ypengbk/litho/full_litho/Xtrain5_3" 
for k in range(2700):
    fileDir = "/home/ypengbk/litho/full_litho/iccad5/train/"  
    tarDir = '/home/ypengbk/litho/full_litho/iccad5/trainb6/X5traincom{}/'.format(k)
    if not os.path.isdir(tarDir):
        os.makedirs(tarDir)
    copyFile(fileDir,tarDir)
    os.chdir('/home/ypengbk/litho/full_litho/iccad5/trainb6/X5traincom{}/'.format(k))     
    i=0
    j= 0

    for filename in os.listdir('/home/ypengbk/litho/full_litho/iccad5/trainb6/X5traincom{}/'.format(k)): 

        if i == 0:
            img0 = imageio.imread(filename)
            i = i + 1
        elif i == 1:
            img1 = imageio.imread(filename)
            i = i + 1
        elif i == 2:
            img2 = imageio.imread(filename)
            i = i + 1
        elif i == 3:
            img3 = imageio.imread(filename)
            i = i + 1
        elif i == 4:  # number of images selected
            break
    for filename in os.listdir('/home/ypengbk/litho/full_litho/iccad5/trainb6/X5traincom{}/'.format(k)):
        if "NHS" in filename:
            j = j
        else:
            j = j + 1
    num.append(j)

    if j == 0:
        y.append(0)
    else:
        y.append(1)
    imgh = np.zeros([1200, 300], np.uint8)
    imgv = np.zeros([300, 2700], np.uint8)
    img_tmp1 = np.hstack((img0, imgh))
    img_tmp2 = np.hstack((img_tmp1, img1))
    img_tmp3 = np.vstack((img_tmp2, imgv))

    img_tmp4 = np.hstack((img2, imgh))
    img_tmp5 = np.hstack((img_tmp4, img3))
    img = np.vstack((img_tmp3, img_tmp5))

    print("shape", img0.shape)
    print("shape",img.shape)
    print("shape",img_tmp3.shape)
    print("shape", img_tmp5.shape)
    img = resize(img, [600, 600]) 
    img = (img * 255).astype('uint8')

    print("shape", img.shape)
    plt.imsave(os.path.join(dest_dir, "{}_disp.png".format(lxy)), img, cmap='gray')
    print('photo {} finished'.format(lxy))
    x.append(img)
    k = k +1
    lxy = lxy +1

temp1="home/ypengbk/litho/full_litho/Comdata6/" # the address of your datasets
if not os.path.isdir(temp1):
    os.makedirs(temp1)
np.save('/home/ypengbk/litho/full_litho/Comdata6/Xtrain5.npy', x)
np.save("/home/ypengbk/litho/full_litho/Comdata6/Ytrain5.npy", y)
np.save("/home/ypengbk/litho/full_litho/Comdata6/Ntrain5.npy", num)