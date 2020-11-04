import numpy as np
from collections import OrderedDict
import os
from torch.autograd import Variable
import glob
import torch
import cv2
import torch.utils.data as data
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision.models import resnet50
from sklearn.ensemble import IsolationForest  
from sklearn.neighbors import LocalOutlierFactor
from utils import *
from torchvision.models import resnet50
from sklearn.metrics import roc_auc_score
from model import RobustPCC as rp
from model import recon_error_pca as rep
rng = np.random.RandomState(2020)
# using ResNeSt-50 as an example
# from resnest.torch import resnest50

def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized

def get_Labels_list(test_folder = "X:\\Anomaly_Dataset\\UCSD_Anomaly_Dataset.v1p2\\ped2\\training\\frames",labels=None):
    '''
    根据labels 获取labels_list 每个视频都会少4帧，因为是预测未来帧模型
    '''
    labels_list = []
    label_length = 0
    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    for video in videos_list:
        video_name = video.split('\\')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    for video in sorted(videos_list):
        video_name = video.split('\\')[-1]
        labels_list = np.append(labels_list, labels[0][4+label_length:videos[video_name]['length']+label_length])
        label_length += videos[video_name]['length']
    return labels_list




def IForestDetect(video_folder,labels_path,nor_ratio=0.2,abnor_ratio=0.01):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    
    # model = resnest50(pretrained=False)
    # model.cuda()
    # model.load_state_dict(torch.load("X:\\Downloads\\resnest50-528c19ca.pth"))

    model = resnet50(pretrained=False)
    model.cuda()
    model.load_state_dict(torch.load("X:\\Downloads\\resnet50-19c8e357.pth"))

    ped2 = PedDataset(video_folder,None,256,256) 
    model.eval()
    imgs = [] 
    feature_1000 = None
    flag = True
    with torch.no_grad():
        for idx, imagePath in enumerate(ped2.samples):
            img = cv2.imread(imagePath)
            img = np.transpose(img,(2,0,1))
            imgs.append(img)
            if (idx+1) % 32 == 0:
                batch = np.asarray(imgs)
                batch = torch.tensor(batch,dtype=torch.float32)
                batch = Variable(batch).cuda()
                if flag:
                    feature_1000 = model.forward(batch)
                    flag = False
                else:
                    xx = model.forward(batch)
                    feature_1000 = torch.cat((feature_1000,xx),0)
                imgs = []
                batch = None
    
        if imgs != []:
            batch = np.asarray(imgs)
            batch = torch.tensor(batch,dtype=torch.float32)
            batch = Variable(batch).cuda()
            feature_1000 = torch.cat((feature_1000,model(batch)),0)
    
    x = feature_1000.cpu().detach().numpy()
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components = 100)
    principalComponents = pca.fit_transform(x)

    labels = np.load(labels_path)
    # if isTrain:
    #     test1_label = loadTestLabel('./data/UCSDped2_train.txt')
    #     test2_label = loadTestLabel('./data/UCSDped2_test.txt')
    #     labels = np.concatenate((test1_label,test2_label),1)

    ########################## get labels_list for labels ######################
    # train_folder = "X:\\Anomaly_Dataset\\UCSD_Anomaly_Dataset.v1p2\\ped2\\training\\frames"
    labels_list = get_Labels_list(video_folder,labels)
    
    ################# IForest 分数越小于0，越有可能是异常值 ###################
    clf = IsolationForest(n_estimators=100, contamination=0.3, random_state=2020, n_jobs=-1,bootstrap=True)   
    y_pred_train = clf.fit_predict(principalComponents)
    IForest_scores_pred = clf.decision_function(principalComponents) 
    IForest_maxmin_normal = anomaly_score(IForest_scores_pred,np.max(IForest_scores_pred),np.min(IForest_scores_pred))
    IForest_maxmin_normal = 1-IForest_maxmin_normal
    # roc = roc_auc_score(labels_list[-1962:],IForest_maxmin_normal[-1962:])
    # print('IForest_nor_rotio',get_nor_rotio(labels_list,-IForest_scores_pred,nor_ratio))
    # print('IForest_abnor_rotio',get_abnor_rotio(labels_list,-IForest_scores_pred,abnor_ratio))

    print('maxmin_IForest_nor_rotio',get_nor_rotio(labels_list,IForest_maxmin_normal,nor_ratio))
    print('maxmin_IForest_abnor_rotio',get_abnor_rotio(labels_list,IForest_maxmin_normal,abnor_ratio))


    ############## PCA_Recon_Error ############
    pre = rep.PCA_Recon_Error(principalComponents, contamination=0.3,random_state=2020)
    pre_result = pre.get_anomaly_score()
    # print('rep_nor_rotio',get_nor_rotio(labels_list,pre_result,nor_ratio))
    # print('rep_abnor_rotio',get_abnor_rotio(labels_list,pre_result,abnor_ratio))

    rep_maxmin_normal = anomaly_score(pre_result,np.max(pre_result),np.min(pre_result))
    print('maxmin_rep_nor_rotio',get_nor_rotio(labels_list,rep_maxmin_normal,nor_ratio))
    print('maxmin_rep_abnor_rotio',get_abnor_rotio(labels_list,rep_maxmin_normal,abnor_ratio))

    ############# IForest PCA #######################
    IForset_PCA_scores = rep_maxmin_normal+IForest_maxmin_normal
    #IForset_PCA_scores = rep_maxmin_normal
    print('Reliability of normal representatives',get_nor_rotio(labels_list,IForset_PCA_scores,nor_ratio))
    print('Reliability of abnormal representatives',get_abnor_rotio(labels_list,IForset_PCA_scores,abnor_ratio))


    #################### 调试使用 ################
    
    # roc_2 = roc_auc_score(labels_list[-1962:],maxmin_normal[-1962:])

    

    select_len_abnor = int(abnor_ratio*len(IForset_PCA_scores))
    indices_abnor = (np.argsort(IForset_PCA_scores)[-select_len_abnor:])   
    #indices_abnor = np.where(labels_list == 1)[0]

    select_len_nor = int(nor_ratio*len(IForset_PCA_scores))
    indices_nor = (np.argsort(IForset_PCA_scores)[:select_len_nor])
    count = 0
    for index in indices_nor:
        if labels_list[index] ==0:

            count += 1
        # else:
        #     #print(index)
    nor_rotio = count/select_len_nor
    print("Reliability of normal representatives",nor_rotio)

    count = 0
    for index in indices_abnor:
        if labels_list[index] ==1:
            count += 1
        # else:
        #     #print(index)
    abnor_rotio = count/select_len_abnor
    print("Reliability of abnormal representatives",abnor_rotio)
    
    ## 人为参与，筛选正常和异常数据
    human_indices_nor = indices_nor[np.where(labels_list[indices_nor]==0)[0]]
    human_indices_abnor = indices_abnor[np.where(labels_list[indices_abnor]==1)[0]]
    #res_nor_ratio = normal_ratio(indices_nor,labels_list)
    
    

    ##################################################

    model.zero_grad()
    #np.save("./data/"+"ped1_human_50%indices_nor.npy",human_indices_nor)
    #np.save("./data/"+"avenue_human_indices_abnor.npy",human_indices_abnor)
    return indices_nor,indices_abnor
   
def get_abnor_rotio(labels_list,scores_pred,rotio=0.05):
    '''
    查看伪标签（abnor）的正确率/ 假阳率（False positive rate）：FPR = FP/(FP+TN)  负样本中，被识别为真的概率
    Reliability of abnormal representatives
    '''
    indices_desc = np.argsort(-scores_pred)
    anomaly_num = int(len(scores_pred) * rotio)
    anomaly_indices = indices_desc[:anomaly_num]
    return np.sum(labels_list[anomaly_indices] == 1)/anomaly_num


def get_nor_rotio(labels_list,scores_pred,rotio=0.2):
    '''
    查看伪标签（nor）的正确率/真阳率（True positive rate）：TPR = TP/（TP+FN）  正样本中，能被识别为真的概率
    Reliability of normal representatives
    '''
    indices_ascend = np.argsort(scores_pred)
    anomaly_num = int(len(labels_list) * rotio)
    anomaly_indices = indices_ascend[:anomaly_num]
    return np.sum(labels_list[anomaly_indices] == 0)/anomaly_num

class PedDataset(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples()
        
        
    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('\\')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
            
            
    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('\\')[-1]
            for i in range(len(self.videos[video_name]['frame'])-self._time_step):
                frames.append(self.videos[video_name]['frame'][i])
                           
        return frames               
            
        
    def __getitem__(self, index):
        video_name = self.samples[index].split('\\')[-2]
        frame_name = int(self.samples[index].split('\\')[-1].split('.')[-2])
        
        batch = []
        for i in range(self._time_step+self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name+i], self._resize_height, self._resize_width)
            if self.transform is not None:
                batch.append(self.transform(image))

        return np.concatenate(batch, axis=0)
        
        
    def __len__(self):
        return len(self.samples)

class TrainDataset(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, indices, time_step=4, num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples(indices)
        
        
    def setup(self):
        videos_n = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos_n):
            video_name = video.split('\\')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
            
            
    def get_all_samples(self,indices):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('\\')[-1]
            for i in range(len(self.videos[video_name]['frame'])-self._time_step):
                frames.append(self.videos[video_name]['frame'][i])

        frames = np.asarray(frames)    
        return frames[indices]               
            
        
    def __getitem__(self, index):
        video_name = self.samples[index].split('\\')[-2]
        frame_name = int(self.samples[index].split('\\')[-1].split('.')[-2])
        
        batch = []
        for i in range(self._time_step+self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name+i], self._resize_height, self._resize_width)
            if self.transform is not None:
                batch.append(self.transform(image))

        return np.concatenate(batch, axis=0)
        
        
    def __len__(self):
        return len(self.samples)
