# -*- coding: UTF-8 -*-
# author : Zhen Qian 2020-08
# prepare theta-phi position map for JUNO center dectector
#   

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns  # for making plots with seaborn
import math

'''
输出的图像的尺寸.
'''
nx = 256 # x axis pixel num
ny = 256 # y axis pixel num

'''
读入的PMT文件;
输出的地图文件.
'''
pos_file="./PMTPos_Acrylic_with_chimney.csv"
out_map = "id2pos_theta-phi_"+str(nx)+"-"+str(ny)+".txt"


def make_map():

    '''
    读入PMT位置.
    '''
    pmt_pos = pd.read_csv(pos_file,header=None,sep=" ")

    pmt_id = pmt_pos.iloc[:,0]
    pmt_x = pmt_pos.iloc[:,1]
    pmt_y = pmt_pos.iloc[:,3]
    pmt_z = pmt_pos.iloc[:,5]
    pmt_theta = pmt_pos.iloc[:,7]
    pmt_theta = pmt_theta*math.pi/180. #pi
    pmt_phi = pmt_pos.iloc[:,8]
    pmt_phi = pmt_phi*math.pi/180. #2pi


    '''
    准备两个通道, 一个放id, 一个放overlap数目, 可以用于画图检查PMT在图片上的排布.
    '''
    pos2map = np.zeros([nx,ny,2], dtype=float)
    pos2map[:,:,0] = -1  #0也是id, 避免冲突, 所以id初始化为-1


    '''
    准备一个列表, 作为id到图片像素的映射, 这个文件将会最终输出, 用于图片映射.
    '''
    id2map = np.zeros([pmt_x.shape[0],3], dtype=int) - 1


    '''
    遍历所有PMT, 将theta-phi映射到图片的x, y
    '''
    for i in range(pmt_x.shape[0]):
        y = theta2y(pmt_theta[i])
        x = phi2x(pmt_phi[i])
        pos2map[x,y,0] =  pmt_id[i]
        pos2map[x,y,1] += 1

        id2map[i,0] = pmt_id[i]
        id2map[i,1] = x
        id2map[i,2] = y


    '''
    画图检查
    '''
    findoverlap(pos2map)
    pos2map = pos2map.transpose(1,0,2)
    plot_map(pos2map)
    
    '''
    输出map文件
    '''
    df = pd.DataFrame(id2map)
    df.to_csv(out_map, index=False, header=False)


def phi2x(phi):
    '''
    get the position from value of phi in x axis
    '''
    global nx
    x = phi/(2*math.pi)*nx
    return int(x)

def theta2y(theta):
    '''
    get the position from value of theta in y axis
    '''
    global ny
    y = theta/(math.pi)*ny
    return int(y)

def findoverlap(pos2map):
    '''
    统计有多少像素是0,是重叠1,还是重叠2......
    '''
    num_0 = 0
    num_1 = 0
    num_2 = 0
    num_3 = 0
    num_l3 = 0
    
    overlap = pos2map.reshape(-1,2)
    #print overlap[10000:1010,:]
    for i in range(overlap.shape[0]):
        if overlap[i,1] == 0:
            num_0 += 1
        elif overlap[i,1] == 1:
            num_1 += 1
        elif overlap[i,1] == 2:
            num_2 += 1
        elif overlap[i,1] == 3:
            num_3 += 1
        elif overlap[i,1] > 3:
            num_l3 += 1
        else:
            print("error: please check!")
    print(' pixels of 0 pmt: %d\n pixels of 1 pmt: %d\n pixels of 2 pmt: %d\n pixels of 3 pmt: %d\n pixels of >3 pmt: %d' %(num_0,num_1,num_2,num_3,num_l3))

def plot_map(pos2map):
    '''
    画出ID的图像和PMT overlap的图像, 用于检查.
    '''
    xspace = 20 # interval
    yspace = 20 
    fontsize=12 #label

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,6))
    maskarr = np.ma.masked_where(pos2map[:,:,0] < 0, pos2map[:,:,0])
    sns.heatmap(pos2map[:,:,0], cmap="coolwarm", mask=maskarr.mask, cbar_kws={'label':'pmtID', "shrink": 1}, xticklabels=xspace, yticklabels=yspace, ax=ax1)
    sns.heatmap(pos2map[:,:,1], cmap="jet", mask=maskarr.mask, cbar_kws={'label':'PMT Overlap', "shrink": 1}, xticklabels=xspace, yticklabels=yspace, ax=ax2)
    ax1.tick_params(labelsize=fontsize)
    ax2.tick_params(labelsize=fontsize)
    ax1.set_title('pmtID',weight='bold', fontsize=18)
    ax2.set_title('PMT Overlap',weight='bold', fontsize=18)
    
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=45)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=45)

    ax1.set_xlabel('phi')
    ax1.set_ylabel('theta')
    ax2.set_xlabel('phi')
    ax2.set_ylabel('theta')
    fig.suptitle('theta-phi map with '+str(nx)+'*'+str(ny)+'pixels')
    #plt.show()
    fig.savefig('theta-phi/id2pos_theta-phi_'+str(nx)+'-'+str(ny)+'.png', dpi=100)
    

make_map()