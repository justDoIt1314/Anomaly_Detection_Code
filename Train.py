import numpy as np
import os
import sys
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
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
from model.dataset import PedDataset,TrainDataset,Initial_Anomaly_Detection,get_Labels_list,get_nor_rotio,get_abnor_rotio
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
import glob

import argparse


parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs for training')
parser.add_argument('--normal_scale', type=float, default=0.2, help='Scales of Normal Representatives')
parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped1, ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset/', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
parser.add_argument('--model_dir', type=str, default='.\\exp\\ped2\\log\\model.pth',help='directory of model')
parser.add_argument('--m_items_dir', type=str, default='.\\exp\\ped2\\log\\keys.pt',help='directory of model')
parser.add_argument('--test_label', type=str, default='./data/avenue_test.txt',help='directory of model, .txt')
parser.add_argument('--train_label', type=str, default='./data/avenue_train.txt',help='directory of model, .txt')
parser.add_argument('--labels', type=str, default='./data/train_and_test_labels_ped2.npy',help='train_and_test_labels_ped2,frame_labels_avenue.npy, directory of model, .npy')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

def get_nor_abnor_RecontructLoss_by_model(train_folder,indices_nor,indices_abnor,model,m_items):
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
            mse_imgs = torch.mean(loss_func_mse(outputs, imgs[:,12:])).item()
            mse_feas = compactness_loss.item()
            abnor_mse_imgs_append = np.append(abnor_mse_imgs_append,mse_imgs)
            abnor_mse_feas_append = np.append(abnor_mse_feas_append,mse_feas)

    with torch.no_grad():
        for k,(imgs) in enumerate(nor_train_loader):
            imgs = Variable(imgs).cuda()     
            outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)
            mse_imgs = torch.mean(loss_func_mse(outputs, imgs[:,12:])).item()
            mse_feas = compactness_loss.item()
            nor_mse_imgs_append = np.append(nor_mse_imgs_append,mse_imgs)
            nor_mse_feas_append = np.append(nor_mse_feas_append,mse_feas)
    print("rec abnor/nor: ",np.mean(abnor_mse_imgs_append)/np.mean(nor_mse_imgs_append))
    
    nor_rec_loss,abnor_rec_loss = np.mean(nor_mse_imgs_append),np.mean(abnor_mse_imgs_append)
    return nor_rec_loss,abnor_rec_loss
    return torch.from_numpy(nor_rec_loss),torch.from_numpy(abnor_rec_loss)


def train_all_epochs(epochs,indices,train_folder,test_folder,indices_nor,indices_abnor):
    '''
    indices:
    indices_nor:
    indices_abnor:
    '''
    
    train_dataset = TrainDataset(train_folder, transforms.Compose([
                transforms.ToTensor(),          
                ]), resize_height=args.h, resize_width=args.w, indices=indices, time_step=args.t_length-1)
    train_dataloader = data.DataLoader(train_dataset, batch_size = args.batch_size, 
                                shuffle=True, num_workers=args.num_workers, drop_last=False)
    # Model setting
    model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)
    params_encoder =  list(model.encoder.parameters()) 
    params_decoder = list(model.decoder.parameters())
    params = params_encoder + params_decoder
    optimizer = torch.optim.Adam(params, lr = args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)
    model.cuda()
    # Report the training process

    log_dir = os.path.join('./exp', args.dataset_type, args.exp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # orig_stdout = sys.stdout
    # f = open(os.path.join(log_dir, 'log.txt'),'w')
    # sys.stdout= f
    loss_func_mse = nn.MSELoss(reduction='none')
    # Training

    m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda() # Initialize the memory items
    len_train_batch = len(train_dataloader)

    model_path = os.path.join(log_dir, 'model.pth')
    keys_path = os.path.join(log_dir, 'keys.pt')
    max_auc = 0
    for epoch in range(epochs):   
        model.eval()
        nor_threshold,abnor_threshold = get_nor_abnor_RecontructLoss_by_model(test_folder,indices_nor,indices_abnor,model,m_items)
        model.train()    
        my_rec_scores = []
        for j,(imgs) in enumerate(train_dataloader):
            
            imgs = Variable(imgs).cuda()
            
            outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(imgs[:,0:12], m_items, True)        
            optimizer.zero_grad()

            rec_losses = loss_func_mse(outputs, imgs[:,12:])
            loss_pixel = torch.mean(rec_losses)
            list_mse_loss = []
            for rec_loss in rec_losses:
                list_mse_loss.append(torch.mean(rec_loss))

            scores = []
            for score in list_mse_loss:
                score = score.cpu().detach().numpy()
                if score <= nor_threshold:
                    scores.append(0.2*score/nor_threshold)
                elif score >= abnor_threshold:
                    scores.append(1)
                else:
                    scores.append(0.2+0.8*(score-nor_threshold)/(abnor_threshold-nor_threshold))
            my_rec_scores.extend(scores)
            loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
            writer.add_scalar("loss_pixel", loss_pixel.item(),j + epoch * len_train_batch)
            writer.add_scalar("compactness_loss", compactness_loss.item(),j + epoch * len_train_batch)
            writer.add_scalar("separateness_loss", separateness_loss.item(),j + epoch * len_train_batch)
            writer.add_scalar("loss", loss.item(), j + epoch * len_train_batch)
            if j % 5 == 0:
                print("epoch:{0} batch {1}/{2} loss:{3} score:{4}".format(epoch,j,len_train_batch,loss.item(),scores))
            loss.backward(retain_graph=True)
            optimizer.step()
        my_rec_scores = np.asarray(my_rec_scores)

        scheduler.step()
        
        print('----------------------------------------')
        print('Epoch:', epoch+1)
        print('Loss: Reconstruction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(loss_pixel.item(), compactness_loss.item(), separateness_loss.item()))
        print('Memory_items:')
        print(m_items)       
        print('----------------------------------------')

        auc = evaluate(test_folder,model_path,keys_path,nor_threshold,abnor_threshold,model=model,m_items=m_items,isBymodel=True)
        
        print(auc)
        if auc > max_auc:
            max_auc = auc
            torch.save(model, os.path.join(log_dir, 'model'+'_'+str(epoch)+'.pth'))
            torch.save(m_items, os.path.join(log_dir, 'keys'+'_'+str(epoch)+'.pt'))
        if epoch > epochs-3:
            torch.save(model, os.path.join(log_dir, 'model'+'_'+str(epoch)+'.pth'))
            torch.save(m_items, os.path.join(log_dir, 'keys'+'_'+str(epoch)+'.pt'))

    model.zero_grad()
    
def Self_train_all_epochs(epochs,indices,train_folder,test_folder,indices_nor,indices_abnor):
    '''
    indices:
    indices_nor:
    indices_abnor:
    '''
    
    
    # Model setting
    model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)
    params_encoder =  list(model.encoder.parameters()) 
    params_decoder = list(model.decoder.parameters())
    params = params_encoder + params_decoder
    optimizer = torch.optim.Adam(params, lr = args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)
    model.cuda()
    # Report the training process

    log_dir = os.path.join('./exp', args.dataset_type, args.exp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # orig_stdout = sys.stdout
    # f = open(os.path.join(log_dir, 'log.txt'),'w')
    # sys.stdout= f
    loss_func_mse = nn.MSELoss(reduction='none')
    # Training

    m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda() # Initialize the memory items
    

    model_path = os.path.join(log_dir, 'model.pth')
    keys_path = os.path.join(log_dir, 'keys.pt')
    max_auc = 0
    for epoch in range(epochs):   

        train_dataset = TrainDataset(train_folder, transforms.Compose([
                transforms.ToTensor(),          
                ]), resize_height=args.h, resize_width=args.w, indices=indices, time_step=args.t_length-1)
        train_dataloader = data.DataLoader(train_dataset, batch_size = args.batch_size, 
                                shuffle=True, num_workers=args.num_workers, drop_last=False)
        len_train_batch = len(train_dataloader)
        model.eval()
        nor_threshold,abnor_threshold = get_nor_abnor_RecontructLoss_by_model(test_folder,indices_nor,indices_abnor,model,m_items)
        model.train()    
        my_rec_scores = []
        for j,(imgs) in enumerate(train_dataloader):
            
            imgs = Variable(imgs).cuda()
            
            outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(imgs[:,0:12], m_items, True)        
            optimizer.zero_grad()

            rec_losses = loss_func_mse(outputs, imgs[:,12:])
            loss_pixel = torch.mean(rec_losses)
            list_mse_loss = []
            for rec_loss in rec_losses:
                list_mse_loss.append(torch.mean(rec_loss))

            scores = []
            for score in list_mse_loss:
                score = score.cpu().detach().numpy()
                if score <= nor_threshold:
                    scores.append(0.2*score/nor_threshold)
                elif score >= abnor_threshold:
                    scores.append(1)
                else:
                    scores.append(0.2+0.8*(score-nor_threshold)/(abnor_threshold-nor_threshold))
            my_rec_scores.extend(scores)
            loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
            writer.add_scalar("loss_pixel", loss_pixel.item(),j + epoch * len_train_batch)
            writer.add_scalar("compactness_loss", compactness_loss.item(),j + epoch * len_train_batch)
            writer.add_scalar("separateness_loss", separateness_loss.item(),j + epoch * len_train_batch)
            writer.add_scalar("loss", loss.item(), j + epoch * len_train_batch)
            if j % 5 == 0:
                print("epoch:{0} batch {1}/{2} loss:{3} score:{4}".format(epoch,j,len_train_batch,loss.item(),scores))
            loss.backward(retain_graph=True)
            optimizer.step()
        my_rec_scores = np.asarray(my_rec_scores)

        scheduler.step()
        
        print('----------------------------------------')
        print('Epoch:', epoch+1)
        print('Loss: Reconstruction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(loss_pixel.item(), compactness_loss.item(), separateness_loss.item()))
        print('Memory_items:')
        print(m_items)       
        print('----------------------------------------')

        indices,indices_nor,indices_abnor, auc = Self_evaluate(test_folder,model_path,keys_path,nor_threshold,abnor_threshold,model=model,m_items=m_items,isBymodel=True,epo=epoch)
        
        print(auc)
        if auc > max_auc:
            max_auc = auc
            torch.save(model, os.path.join(log_dir, 'self_model'+'_'+str(epoch)+'.pth'))
            torch.save(m_items, os.path.join(log_dir, 'self_keys'+'_'+str(epoch)+'.pt'))
            printLog(os.path.join(log_dir, 'log.txt'),"self_epoch_"+str(epoch)+" auc",auc)
            
        if epoch > epochs-3:
            torch.save(model, os.path.join(log_dir, 'self_model'+'_'+str(epoch)+'.pth'))
            torch.save(m_items, os.path.join(log_dir, 'self_keys'+'_'+str(epoch)+'.pt'))
            printLog(os.path.join(log_dir, 'log.txt'),"self_epoch_"+str(epoch)+" auc",auc)

    model.zero_grad()

def Self_evaluate(test_folder, model_dir, m_items_dir, nor_threshold, abnor_threshold, model=None,m_items=None,isBymodel=False,epo=0):
    log_dir = os.path.join('./exp', args.dataset_type, args.exp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Loading dataset
    test_dataset = PedDataset(test_folder, transforms.Compose([
                transforms.ToTensor(),            
                ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

    test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                                shuffle=False, num_workers=args.num_workers_test, drop_last=False)

    loss_func_mse = nn.MSELoss(reduction='none')

    # Loading the trained model
    
    if not isBymodel:
        model = torch.load(model_dir)
        model.cuda()
        m_items = torch.load(m_items_dir)

    labels = np.load(args.labels)
    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
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
        for k,(imgs) in enumerate(test_batch):

            if k == label_length-4*(video_num+1):
                video_num += 1
                label_length += videos[videos_list[video_num].split('\\')[-1]]['length']

            imgs = Variable(imgs).cuda()           
            outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)
            #mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
            mse_imgs = torch.mean(loss_func_mse(outputs, imgs[:,12:])).item()
            mse_feas = compactness_loss.item()
            mse_imgs_append = np.append(mse_imgs_append,mse_imgs)
            # mse_feas_append = np.append(mse_feas_append,mse_feas)

            # Calculating the threshold for updating at the test time
            # point_sc = point_score(outputs, imgs[:,3*4:])

            # if  point_sc < args.th:
            #     query = F.normalize(feas, dim=1)
            #     query = query.permute(0,2,3,1) # b X h X w X d
            #     m_items_test = model.memory.update(query, m_items_test, False)

            psnr_list[videos_list[video_num].split('\\')[-1]].append(psnr(mse_imgs))
            feature_distance_list[videos_list[video_num].split('\\')[-1]].append(mse_feas)


    ######################### 结合异常数据算得分  #####################
    my_rec_score = update_anomaly_score(mse_imgs_append,nor_threshold,abnor_threshold)    
    # #anomaly_score_total_list = my_rec_score*args.alpha+my_compact_score*(1-args.alpha)
    my_accuray = roc_auc_score(labels_list,my_rec_score)
    if args.dataset_type == 'ped2':
        auc_test = roc_auc_score(labels_list[-1962:],my_rec_score[-1962:])
    elif args.dataset_type == 'ped1':
        auc_test = roc_auc_score(labels_list[-7056:],my_rec_score[-7056:])
    else:
        auc_test = my_accuray
    print("my_accuracy:",my_accuray*100,'%')
    print("auc_test:",auc_test*100,'%')
    # Measuring the abnormality score and the AUC
    
    nor_ratio = 0.2+0.01*epo
    abnor_ratio = 0.01
    TPR = get_nor_rotio(labels_list,my_rec_score,nor_ratio)
    FPR = get_abnor_rotio(labels_list,my_rec_score,abnor_ratio)
    print("TPR:",TPR)
    print("FPR",FPR)


    select_len_abnor = int(abnor_ratio*len(my_rec_score))
    indices_abnor = (np.argsort(my_rec_score)[-select_len_abnor:])   
    #indices_abnor = np.where(labels_list == 1)[0]

    select_len_nor = int(nor_ratio*len(my_rec_score))
    indices_nor = (np.argsort(my_rec_score)[:select_len_nor])
    ######################### min-max 得分机制 #######################
    # anomaly_score_total_list = []
    # for video in sorted(videos_list):
    #     video_name = video.split('\\')[-1]
    #     anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]), 
    #                                     anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)

    # anomaly_score_total_list = np.asarray(anomaly_score_total_list)

    # accuracy = AUC(1-anomaly_score_total_list, np.expand_dims(labels_list, 0))
    # print('Mem The result of ', args.dataset_type)
    # print('Mem AUC: ', accuracy*100, '%')
    #################################################################
    f = open(os.path.join(log_dir, 'log.txt'),'a+')
    f.write("AUC: "+str(my_accuray)+" aut_test:"+str(auc_test)+"\n")
    f.write("TPR: "+str(TPR)+" FPR:"+str(FPR)+"\n")
    f.close()
    return indices_nor,indices_nor[:int(len(indices_nor)/4)],indices_abnor,my_accuray
def evaluate(test_folder, model_dir, m_items_dir, nor_threshold, abnor_threshold, model=None,m_items=None,isBymodel=False):
    log_dir = os.path.join('./exp', args.dataset_type, args.exp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Loading dataset
    test_dataset = PedDataset(test_folder, transforms.Compose([
                transforms.ToTensor(),            
                ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

    test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                                shuffle=False, num_workers=args.num_workers_test, drop_last=False)

    loss_func_mse = nn.MSELoss(reduction='none')

    # Loading the trained model
    
    if not isBymodel:
        model = torch.load(model_dir)
        model.cuda()
        m_items = torch.load(m_items_dir)

    labels = np.load(args.labels)
    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
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
        for k,(imgs) in enumerate(test_batch):

            if k == label_length-4*(video_num+1):
                video_num += 1
                label_length += videos[videos_list[video_num].split('\\')[-1]]['length']

            imgs = Variable(imgs).cuda()           
            outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)
            #mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
            mse_imgs = torch.mean(loss_func_mse(outputs, imgs[:,12:])).item()
            mse_feas = compactness_loss.item()
            mse_imgs_append = np.append(mse_imgs_append,mse_imgs)
            # mse_feas_append = np.append(mse_feas_append,mse_feas)

            # Calculating the threshold for updating at the test time
            # point_sc = point_score(outputs, imgs[:,3*4:])

            # if  point_sc < args.th:
            #     query = F.normalize(feas, dim=1)
            #     query = query.permute(0,2,3,1) # b X h X w X d
            #     m_items_test = model.memory.update(query, m_items_test, False)

            psnr_list[videos_list[video_num].split('\\')[-1]].append(psnr(mse_imgs))
            feature_distance_list[videos_list[video_num].split('\\')[-1]].append(mse_feas)


    ######################### 结合异常数据算得分  #####################
    my_rec_score = update_anomaly_score(mse_imgs_append,nor_threshold,abnor_threshold)    
    # #anomaly_score_total_list = my_rec_score*args.alpha+my_compact_score*(1-args.alpha)
    my_accuray = roc_auc_score(labels_list,my_rec_score)
    
    print("my_accuracy:",my_accuray*100,'%')


    #################### 调试使用 ################
    # Measuring the abnormality score and the AUC

    ######################### min-max 得分机制 #######################
    # anomaly_score_total_list = []
    # for video in sorted(videos_list):
    #     video_name = video.split('\\')[-1]
    #     anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]), 
    #                                     anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)

    # anomaly_score_total_list = np.asarray(anomaly_score_total_list)

    # accuracy = AUC(1-anomaly_score_total_list, np.expand_dims(labels_list, 0))
    # print('Mem The result of ', args.dataset_type)
    # print('Mem AUC: ', accuracy*100, '%')
    #################################################################
    f = open(os.path.join(log_dir, 'log.txt'),'a+')
    f.write("AUC: "+str(my_accuray)+"\n")
    f.close()
    return my_accuray

if __name__ == "__main__":
    print("hello")
    writer = SummaryWriter()
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

    train_folder = args.dataset_path+args.dataset_type+"\\training\\frames"
    test_folder = args.dataset_path+args.dataset_type+"\\testing\\frames"
    train_and_test_folder = args.dataset_path+args.dataset_type+"\\train_and_test\\frames"
    target_folder = train_and_test_folder
    isTrain = False
    ratio = 0.02
    human_indices_nor_path = "./data/"+args.dataset_type+"_human_50%indices_nor.npy"
    human_indices_abnor_path = "./data/"+args.dataset_type+"_human_indices_abnor.npy"

    # if os.path.exists(human_indices_nor_path) and os.path.exists(human_indices_abnor_path):
    #     indices_nor = np.load(human_indices_nor_path)
    #     indices_abnor = np.load(human_indices_abnor_path)
    # else:
    
    # get representatives of normal and abnormal data
    indices_nor,indices_abnor = IForestDetect(target_folder,args.labels,args.normal_scale,0.01)
    
    print("hello")
    log_dir = os.path.join('./exp', args.dataset_type, args.exp_dir)

    # training
    train_all_epochs(args.epochs,indices_nor,target_folder, target_folder,indices_nor[:int(len(indices_nor)/4)],indices_abnor)
    #Self_train_all_epochs(args.epochs,indices_nor,target_folder, target_folder,indices_nor[:int(len(indices_nor)/4)],indices_abnor)
    
    #indices = evaluate(video_folder,model_path,keys_path, isTrain, iter_ratios[idx]) 

    #for epoch in range(30):
    # model_path = os.path.join(log_dir,"model_"+str(epoch)+'.pth')
    # keys_path = os.path.join(log_dir,"keys_"+str(epoch)+'.pt')
    # auc = evaluate(test_folder,model_path,keys_path,False)
    # print("epoch_"+str(epoch)+" :"+str(auc))
    
        
    # sys.stdout = orig_stdout
    # f.close()



