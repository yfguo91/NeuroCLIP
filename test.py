from spikingjelly.datasets.cifar10_dvs  import CIFAR10DVS
import spikingjelly
import torch
root_dir = './data/cifar10dvs'
#train_data = CIFAR10DVS(root_dir, train=True, data_type='frame', frames_number=10, split_by='number')
#test_data = CIFAR10DVS(root_dir, train=False, data_type='frame', frames_number=10, split_by='number')
data_set = CIFAR10DVS(root_dir, data_type='frame', frames_number=10, split_by='number')
train_data, test_data = spikingjelly.datasets.split_to_train_test_set(0.9, data_set, 10, random_split = False)
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import cv2
import numpy as np

def mv_proj(img):
    t, c, w, h = img.shape
    #print(img.shape)
    #print(img[1,0,:,:].max())
    cv2.imshow('2', img[1,0,:,:])
    cv2.waitKey(100)
    #print(img.mean())
    img = img[:,0:1,:,:].repeat(3,axis=1)
    img = torch.tensor(img)
   
    #exit()
    img = torch.nn.functional.upsample(img, size=(224, 224), mode='bilinear', align_corners=True)
    return img

for i, data in enumerate(test_data):
    img = mv_proj(data[0])[3,:,:,:]
    print(data[1])
    img = img.permute(1,2,0)
    img = np.array(img)
    #print(img.max())
    cv2.imshow('test', img)
    cv2.waitKey(1)
    #plt.imshow(img) # 显示图片
    #plt.axis('off') # 不显示坐标轴
    #plt.show()
    
