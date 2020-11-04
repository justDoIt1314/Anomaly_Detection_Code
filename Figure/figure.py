import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator


def plot_sample():
    ped2_nor_sample_len = np.load('ped2_nor_sample_len.npy')
    ped2_nor_len = np.load('ped2_nor_len.npy')
    


    x_values = [i for i in range(1,9)]

    plt.plot(x_values, ped2_nor_sample_len, 'ro-', color='red', alpha=0.8, linewidth=1, label='Sampling normal frames')
    plt.plot(x_values, ped2_nor_len, 'bs-', color='blue', alpha=0.8, linewidth=1, label='Normal frames')


    # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc="upper right")
    plt.xlabel('group')
    plt.ylabel('Number of frames')
    x_major_locator=MultipleLocator(1)
    #把x轴的刻度间隔设置为1，并存在变量里
    #y_major_locator=MultipleLocator(5)
    #把y轴的刻度间隔设置为10，并存在变量里
    ax=plt.gca()
    #ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    #把x轴的主刻度设置为1的倍数
    #ax.yaxis.set_major_locator(y_major_locator)
    #把y轴的主刻度设置为10的倍数
    #plt.ylim(0,70)
    #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.show()
def plot_representative_rate():
    labels_ped1 = np.load('ped1_normal_ratio.npy')
    labels_ped2 = np.load('ped2_normal_ratio.npy')
    labels_avenue = np.load('avenue_normal_ratio.npy')


    x_values = [i for i in range(1,11)]

    plt.plot(x_values, labels_ped1*100, 'ro-', color='red', alpha=0.8, linewidth=1, label='ped1')
    plt.plot(x_values, labels_ped2*100, 'bs-', color='blue', alpha=0.8, linewidth=1, label='ped2')
    plt.plot(x_values, labels_avenue*100, 'g^-', color='orange', alpha=0.8, linewidth=1, label='avenue')

    # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc="upper right")
    plt.xlabel('group')
    plt.ylabel('Normal representative rate(%)')
    x_major_locator=MultipleLocator(1)
    #把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator=MultipleLocator(5)
    #把y轴的刻度间隔设置为10，并存在变量里
    ax=plt.gca()
    #ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    #把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    #把y轴的主刻度设置为10的倍数
    plt.ylim(0,70)
    #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.show()

def plot_auc():
    labels_ped1 = np.array([0.7820,0.7847,0.7837,0.7711,0.7751, 0.7919,0.7764,0.7751,0.78,
    0.7809,0.7778,0.7753,0.7784,0.7717,0.7815,0.7786,0.7818,0.7791,0.7790,0.7834,0.7829,0.7839,
    0.7875,0.7862,0.7856,0.7865,0.7888,0.7838,0.7859,0.7885])
    labels_ped2 = np.array([0.9204,0.9329,0.9353,0.9312,0.9370,0.9329,0.9383,0.9342,0.9264,0.9258,0.9267,
    0.9282,0.9131,0.9241,0.9075,0.9192,0.9192,0.9222,0.9188,0.9221,0.9188,0.9221,0.9185,0.9186,0.9217,
    0.9222,0.9212,0.9215,0.9231,0.9225])
    labels_avenue = np.array([0.8508,0.8440,0.8626,0.8496,0.8573,0.8600,0.8610,0.8550,0.8601,0.8594,0.8607,
    0.8630,0.8633,0.8616,0.8611,0.8680,0.8634,0.8604,0.8628,0.8618,0.8633,0.8606,0.8619,0.8614,0.8617,
    0.8628,0.8611,0.8611,0.8634,0.8618
    ])


    x_values = [i for i in range(1,31)]

    plt.plot(x_values, labels_ped1*100, 'ro-', color='red', alpha=0.8, linewidth=1, label='ped1')
    plt.plot(x_values, labels_ped2*100, 'bs-', color='blue', alpha=0.8, linewidth=1, label='ped2')
    plt.plot(x_values, labels_avenue*100, 'g^-', color='orange', alpha=0.8, linewidth=1, label='avenue')

    # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc="upper right")
    plt.xlabel('Epoch')
    plt.ylabel('AUC(%)')
    x_major_locator=MultipleLocator(5)
    #把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator=MultipleLocator(5)
    #把y轴的刻度间隔设置为10，并存在变量里
    ax=plt.gca()
    #ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    #把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    #把y轴的主刻度设置为10的倍数
    plt.ylim(70,100)
    #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.show()
plot_sample()
plot_auc()
# scores = np.load("ped2_scores_2.npy")
# a = []

# for idx,score in enumerate(scores):
#     if idx > 3010:
#         print(idx,score)