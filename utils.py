import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from sklearn.metrics import roc_auc_score
import random


def normal_ratio(indices_nor,labels_list):
    """
    计算normal分布率，数据集分10组，每组中选取的normal占该组normal的比例
    """
    nor_sample_len=[]
    nor_len=[]
    s = set(indices_nor)
    len_labels = len(labels_list)
    res_nor_ratio = []
    for ith in range(8):
        start = int(ith/8*len_labels)
        end = 0
        if ith == 7:
            end = len_labels
        else:
            end = int((ith+1)/8*len_labels)
        ith_indice = np.where(labels_list[start:end] == 0)[0]
        ith_indice = ith_indice + start
        if len(ith_indice) == 0:
            nor_len.append(0)
            nor_sample_len.append(0)
        else:
            inter_indice = s.intersection(set(ith_indice))
            nor_sample_len.append(len(inter_indice))
            nor_len.append(len(ith_indice))
            res_nor_ratio.append(len(inter_indice)/len(ith_indice))
    np.save("ped2_normal_ratio.npy",np.asarray(res_nor_ratio))
    np.save("ped2_nor_sample_len.npy",np.asarray(nor_sample_len))
    np.save("ped2_nor_len.npy",np.asarray(nor_len))
    return res_nor_ratio

        



def loadTestLabel(path):
    lable = []
    fr = open(path)
    lines = fr.readlines()
    for line in lines:
        lineArr = line.strip().split(" ")
        lable.append(lineArr[-1])
    return np.expand_dims(np.array(lable, dtype='float32'),0)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def psnr(mse):

    return 10 * math.log10(1 / mse)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def normalize_img(img):

    img_re = copy.copy(img)
    
    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))
    
    return img_re

def point_score(outputs, imgs):
    
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)
    normal = (1-torch.exp(-error))
    score = (torch.sum(normal*loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)) / torch.sum(normal)).item()
    return score

def update_anomaly_score(scores,nor_threshold,abnor_threshold):
    
    res = []
    for score in scores:
        if score < nor_threshold:
            # res = np.append(res,0.2*score/nor_threshold)
            res = np.append(res,0)
        elif score > abnor_threshold:
            res = np.append(res,1)
        else:
            res = np.append(res,0.2+0.6*(score-nor_threshold)/(abnor_threshold-nor_threshold))
    return res

def decision_function(score,nor_threshold,abnor_threshold):
    _score = 0.2+0.6*(score-nor_threshold)/(abnor_threshold-nor_threshold)
    if _score < 0:
        _score = 0
    elif _score >1:
        _score = 1
    return _score

def update_anomaly_score_2(scores,nor_threshold,abnor_threshold,abnor_max):
    
    res = []
    for score in scores:
        if score <= nor_threshold:
            res = np.append(res,0.25*score/nor_threshold)
        elif score >= abnor_threshold:
            res = np.append(res,0.75+0.25*(score-abnor_threshold)/(abnor_max-abnor_threshold))
        else:
            res = np.append(res,0.25+0.5*(score-nor_threshold)/(abnor_threshold-nor_threshold))
    return res

def update_anomaly_score_3(scores,nor_threshold,abnor_threshold):
    res = []
    for score in scores:
        _score = 0.2+0.6*(score-nor_threshold)/(abnor_threshold-nor_threshold)
        if _score < 0:
            _score = 0
        elif _score >1:
            _score = 1
        res = np.append(res,_score)
    return res

def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr-min_psnr))

def min_max_normalize(nums):
    min_num = np.min(nums)
    max_num = np.max(nums)
    res = []
    for i in nums:
        res = np.append(res,anomaly_score(i,max_num,min_num))
    return res

def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr)))

def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc

def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha*list1[i]+(1-alpha)*list2[i]))
        
    return list_result

def printLog(log_path,flag,log):
    f = open(log_path,'a+')
    f.write(flag+": "+str(log)+"\n")
    f.close()

def randomSample(indices,segments=5):
    '''
    随机采样，分5段，采样概率线性降低
    '''
    indices = np.asarray(indices,np.int32)
    len_indices = len(indices)
    len_segment = len_indices//segments
    res = []
    cur_index = 0
    for i in range(segments,0,-1):
        
        subIndex = None
        if i != 1:
            subIndex = indices[cur_index:cur_index+len_segment]
        else:
            subIndex = indices[cur_index:]
        len_select = int((i/segments)*len(subIndex))
        select_index = np.random.choice(subIndex,len_select,False)
        res = np.append(res, select_index)
        cur_index = cur_index + len_segment
    res = np.asarray(res,dtype=np.int32)
    return res

# def upSample(indices):
#     '''
#     增加置信度高的样本，从而降低伪标签的错误率
#     '''

