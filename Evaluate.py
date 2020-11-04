import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.dataset import PedDataset,IForestDetect,TrainDataset,get_abnor_rotio,get_nor_rotio
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
import glob
import argparse
from Train import evaluate

parser = argparse.ArgumentParser(description="Anomaly_Detection")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.8, help='weight for the anomality score')
parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset/', help='directory of data')
parser.add_argument('--model_dir', type=str, default='.\\exp\\ped2\\20%_model.pth',help='directory of model')
parser.add_argument('--m_items_dir', type=str, default='.\\exp\\ped2\\20%_keys.pt',help='directory of model')
parser.add_argument('--test_label', type=str, default='./data/test_labels_ped1.npy',help='directory of model, .txt')
parser.add_argument('--train_label', type=str, default='./data/avenue_train.txt',help='directory of model, .txt')
parser.add_argument('--labels', type=str, default='./data/train_and_test_labels_ped2.npy',help='frame_labels_avenue,train_and_test_labels_ped2.npy,directory of model, .npy')
args = parser.parse_args()
#m_items = torch.load(args.m_items_dir)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

def get_nor_abnor_RecontructLoss(train_folder,indices_nor,indices_abnor):
    isTrain = True
    ratio = 0.02
    abnor_train_dataset = TrainDataset(train_folder,transforms.Compose([
                transforms.ToTensor(),            
                ]), resize_height=args.h, resize_width=args.w, indices=indices_abnor,time_step=4, num_pred=1)
    
    abnor_train_loader = data.DataLoader(abnor_train_dataset, batch_size = args.test_batch_size, 
                                shuffle=False, num_workers=args.num_workers_test, drop_last=False)
    
    nor_train_dataset = TrainDataset(train_folder,transforms.Compose([
                transforms.ToTensor(),            
                ]), resize_height=args.h, resize_width=args.w, indices=indices_nor,time_step=4, num_pred=1)
    
    nor_train_loader = data.DataLoader(nor_train_dataset, batch_size = args.test_batch_size, 
                                shuffle=False, num_workers=args.num_workers_test, drop_last=False)

    loss_func_mse = nn.MSELoss(reduction='none')

    # Loading the trained model
    model = torch.load(args.model_dir)
    model.cuda()
    m_items = torch.load(args.m_items_dir)
    m_items_test = m_items.clone()
    nor_mse_imgs_append = []
    nor_mse_feas_append = []
    abnor_mse_imgs_append = []
    abnor_mse_feas_append = []
    model.eval()

    with torch.no_grad():
        for k,(imgs) in enumerate(abnor_train_loader):
            imgs = Variable(imgs).cuda()     
            outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)
            #mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
            mse_imgs = torch.mean(loss_func_mse(outputs, imgs[:,12:])).item()
            mse_feas = compactness_loss.item()
            abnor_mse_imgs_append = np.append(abnor_mse_imgs_append,mse_imgs)
            abnor_mse_feas_append = np.append(abnor_mse_feas_append,mse_feas)

    with torch.no_grad():
        for k,(imgs) in enumerate(nor_train_loader):
            imgs = Variable(imgs).cuda()     
            outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)
            #mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
            mse_imgs = torch.mean(loss_func_mse(outputs, imgs[:,12:])).item()
            mse_feas = compactness_loss.item()
            nor_mse_imgs_append = np.append(nor_mse_imgs_append,mse_imgs)
            nor_mse_feas_append = np.append(nor_mse_feas_append,mse_feas)
    print("rec abnor/nor: ",np.mean(abnor_mse_imgs_append)/np.mean(nor_mse_imgs_append))
    
    return np.mean(nor_mse_imgs_append),np.mean(abnor_mse_imgs_append)

def displayAnomalyArea(out,img,mse_imgs,pre_loss,k):
    img = img.numpy()
    pre_loss = pre_loss.cpu().detach().numpy()    
    out = out.cpu().detach().numpy()   
    out = (out+1)*127.5
    pre_img = (img+1)*127.5
    pre_img = np.transpose(pre_img,[1,2,0])
    out = np.transpose(out,[1,2,0])
    pre_loss = np.transpose(pre_loss,[1,2,0])
    cv2.imwrite("X:\\Anomaly_Dataset\\anomaly_rec\\"+args.dataset_type+"\\pred_img\\"+str(k).zfill(5)+".jpg",out)
    cv2.imwrite("X:\\Anomaly_Dataset\\anomaly_rec\\"+args.dataset_type+"\\gt_img\\"+str(k).zfill(5)+".jpg",pre_img)
    ind = np.where(pre_loss>mse_imgs)
    len_n = len(ind[0])
    for i in range(len_n):
        if ind[2][i] == 0:
            pre_img[ind[0][i],ind[1][i],0] = 0
        elif ind[2][i] == 1:
            pre_img[ind[0][i],ind[1][i],1] = 255                                                                                                           
        else:
            pre_img[ind[0][i],ind[1][i],2] = 255

    cv2.imwrite("X:\\Anomaly_Dataset\\anomaly_rec\\"+args.dataset_type+"\\loc_img\\"+str(k).zfill(5)+".jpg",pre_img)


def npy_to_txt(label_path,score_path,labels_list,scores):
    _len = len(labels_list)
    f_label = open(label_path,'a+')
    f_score = open(score_path,'a+')
    
    for i in range(_len):
        f_label.write(str(labels_list[i])+"\n")
        f_score.write(str(scores[i])+"\n")
    
    f_label.close()
    f_score.close()


if __name__ == "__main__":
    
    #evaluate(args.dataset_path+args.dataset_type+"\\training\\frames",args.model_dir, args.m_items_dir, True)





    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
    #test_folder = 'X:\\Anomaly_Dataset\\UCSD_ped1_ped2\\ped2\\training\\frames'
    test_folder = args.dataset_path+args.dataset_type+"\\testing\\frames"
    train_folder = args.dataset_path+args.dataset_type+"\\training\\frames"
    target_folder = train_folder
    # Get the input for the upper and lower branches
    indices_nor,indices_abnor = IForestDetect(target_folder,args.labels,0.05,0.01)
    # Get the average prediction loss  of the upper and lower branches
    nor_threshold,abnor_threshold = get_nor_abnor_RecontructLoss(target_folder,indices_nor,indices_abnor)
    
    # Loading dataset
    test_dataset = PedDataset(target_folder, transforms.Compose([
                transforms.ToTensor(),            
                ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(target_folder, '*')))

    test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                                shuffle=False, num_workers=args.num_workers_test, drop_last=False)

    loss_func_mse = nn.MSELoss(reduction='none')

    # Loading the trained model
    model = torch.load(args.model_dir)
    model.cuda()
    m_items = torch.load(args.m_items_dir)
    #labels_from_npy = np.load('./data/frame_labels_'+args.dataset_type+'.npy')
    # labels = loadTestLabel('./data/avenue_test.txt')
    labels = np.load(args.labels)
    # test1_label = loadTestLabel('./data/UCSDped2_train.txt')
    # test2_label = loadTestLabel('./data/UCSDped2_test.txt')
    # labels = np.concatenate((test1_label,test2_label),1)
    # np.save('./data/train_and_test_labels_'+args.dataset_type+'.npy',labels)
    # label = np.load('./data/train_and_test_labels_'+args.dataset_type+'.npy')
    # print(np.mean(np.abs(label-labels)))
    
    for video in videos_list:
        video_name = video.split('\\')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    labels_list = []
    label_length = 0
    psnr_list = {}
    feature_distance_list = {}

    print('Evaluation of', args.dataset_type)

    # Setting for video anomaly detection
    for video in sorted(videos_list):
        video_name = video.split('\\')[-1]
        labels_list = np.append(labels_list, labels[0][4+label_length:videos[video_name]['length']+label_length])
        label_length += videos[video_name]['length']
        psnr_list[video_name] = []
        feature_distance_list[video_name] = []

    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('\\')[-1]]['length']
    m_items_test = m_items.clone()
    mse_imgs_append = []
    mse_feas_append = []
    model.eval()
    with torch.no_grad():
        for k,(img) in enumerate(test_batch):

            if k == label_length-4*(video_num+1):
                video_num += 1
                label_length += videos[videos_list[video_num].split('\\')[-1]]['length']

            imgs = Variable(img).cuda()
            
            outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)
            err_mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
            mse_imgs = torch.mean(loss_func_mse(outputs, imgs[:,12:])).item()
            # if labels_list[k] == 1:
            # displayAnomalyArea(outputs[0],img[0,12:],250*nor_threshold,loss_func_mse(outputs[0], imgs[0,12:]),k)
            mse_feas = compactness_loss.item()
            mse_imgs_append = np.append(mse_imgs_append,mse_imgs)
            mse_feas_append = np.append(mse_feas_append,mse_feas)
            # Calculating the threshold for updating at the test time
            point_sc = point_score(outputs, imgs[:,3*4:])
            
            if  point_sc < args.th:
                query = F.normalize(feas, dim=1)
                query = query.permute(0,2,3,1) # b X h X w X d
                m_items_test = model.memory.update(query, m_items_test, False)

            psnr_list[videos_list[video_num].split('\\')[-1]].append(psnr(err_mse_imgs))
            feature_distance_list[videos_list[video_num].split('\\')[-1]].append(mse_feas)

    

    ########################### Decision Network  ##################################################
    my_rec_score = update_anomaly_score(mse_imgs_append,nor_threshold,abnor_threshold)    
    my_accuray = roc_auc_score(labels_list,my_rec_score)
    my_accuray_test = roc_auc_score(labels_list[-1962:],my_rec_score[-1962:])
    print("Decision-Network: The result of",args.dataset_type)
    print("Full dataset:")
    print('AUC: ', my_accuray*100, '%')
    print("testing dataset:")
    print('AUC: ', my_accuray_test*100, '%')

    ####################### PD-MaxMin Measuring the abnormality score and the AUC ######################
    anomaly_score_total_list = []
    for video in sorted(videos_list):
        video_name = video.split('\\')[-1]
        anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]), 
                                        anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)

    anomaly_score_total_list = np.asarray(anomaly_score_total_list)

    accuracy = AUC(1-anomaly_score_total_list, np.expand_dims(labels_list, 0))
    accuracy_test = AUC(1-anomaly_score_total_list[-1962:], np.expand_dims(labels_list[-1962:], 0))
    
    print('PD-MaxMin: The result of ', args.dataset_type)
    print("Full dataset:")
    print('AUC: ', accuracy*100, '%')
    print("testing dataset:")
    print('AUC: ', accuracy_test*100, '%')
