
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import cv2

def query_lime(Xi, model, name):
    name=name+'.png'

    def batch_predict(tmp):
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        #print ('tmp.shape', tmp.shape)
        tmp = np.transpose(tmp[0], (2, 0, 1))
        Xi = tmp[0]
        Xi = Xi.astype(np.uint8)

     

        toT = transforms.ToTensor()
        batch = toT(Xi).to(torch.float32)
        batch = torch.reshape(batch, (1, 1, 600, 600))
        #print ('batch sum, max', batch.sum(), batch.max())
        batch = batch.to(device)
        
        logits = model(batch)
        probs = sigmoid(logits).detach().cpu().numpy()[0][0]
        out = np.array([1-probs, probs])
        #print ('probs', probs, logits)
        out = out.reshape(1, -1)
        return out

    #print ('real input size', Xi.shape)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(Xi,
                                             batch_predict, # classification function
                                             top_labels=2,
                                             hide_color=0,
                                             batch_size=1, # necessary, a bug
                                             num_samples=300)
                                             #progress_bar=False) 

    print ('top label', explanation.top_labels[0])
    #temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                        positive_only=True, num_features=10, hide_rest=True)
    print('temp',temp)
    print('mask',mask)
    print('temp.shape',temp.shape)
    print('mask.shape',mask.shape)
    temp1=np.zeros(360000)
    temp1=temp1.reshape(600,600)
    l=mask.shape[0]
    for i in range(l):
        num=[]
        for x in mask[i]:
            if x>0:
                num.append(x)
        
        if len(num)>0:
           for y in range(len(num)-1):
               if y%2==0:
                    mask[i][num[y]:num[y+1]]=255
               
            
        
    xt=mask
    xt1=Xi
    
    x = xt > 0
    xback = xt1 > 0
    x_all = xback + x * 10

    cmap = colors.ListedColormap(['black', 'white'])
    bounds=[0, 0.1, 10] #district/interval
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    cmap1 = colors.ListedColormap(['red'])
    bounds1=[10,100] #district/interval
    norm1 = colors.BoundaryNorm(bounds1, cmap1.N)
    
    x_al0=(x_all>=10)*0.2
    x_al1=((x_all>=10)==0)
    x_al=x_al0+x_al1
    


    plt.imshow(x_all, cmap=cmap, norm=norm)
    plt.imshow(x_all, cmap=cmap1, norm=norm1,alpha=0.1)
    ax = plt.gca()
    #rect=patches.Rectangle((220,220),160,160,edgecolor='red',linewidth=6,facecolor='none',linestyle='dotted')
    #ax.add_patch(rect)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)

    plt.savefig(name, dpi = 300, bbox_inches='tight')
    #plt.plot(name, dpi = 200, bbox_inches='tight')
    plt.close()
    plt.clf()

