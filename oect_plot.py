#!/usr/bin/env python
from array2gif import write_gif
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

def draw_sample(sample1,sample2,sample3):
    plt.clf()
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.imshow(sample1,cmap=plt.cm.gray_r)
    ax2 = fig.add_subplot(222)
    ax2.imshow(sample2,cmap=plt.cm.gray_r)
    ax3 = fig.add_subplot(223)
    ax3.imshow(sample3,cmap=plt.cm.gray_r)
    plt.savefig('sample.png')
    return 0

def draw_accuracy(accuracy_list):
    plt.clf()
    plt.plot(accuracy_list,'r',label='error rate')
    # plt.legend(loc="upper right")
    plt.xlabel('epochs')
    plt.ylabel('error rate')
    plt.savefig('accuracy.png')

def set_working_path():
    os.chdir(os.path.dirname(__file__))
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cmd = 'mkdir' + ' ' + time_str + 'log/'
    os.system(cmd)
    cmd = time_str + 'log/'
    os.system('cp dataset.npz'+ ' ' + cmd)
    os.chdir(cmd)

def write_accuracy_log(error_rate):
    sf = open('Accuracy.log','a+')
    sf.write(str(error_rate) + '%')
    sf.write('\n')
    sf.close()

def write_system_log(trainingset,teacherset,testset,answer):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = 'snn' + '_simulation' + time_str + 'log.ini'
    sf = open(filename,'w+')
    sf.write('learning sample:\n')
    for i in range(trainingset.shape[0]):
        sf.write(str(trainingset[i]))
        sf.write('\n')
        sf.write(str(teacherset[i]))
        sf.write('\n')
    sf.write('testing sample:\n')
    for i in range(testset.shape[0]):
        sf.write(str(testset[i]))
        sf.write('\n')
        sf.write(str(answer[i]))
        sf.write('\n')
    sf.close()

def draw_line_chart(oect_list,list_len):
    plt.clf()
    color_list = ['r','y','g','c','b','m','k','teal','skyblue']
    for i in range(3):
        for j in range(9):
            globals()['g_list'+str(j)+str(i)] = list()
            for x in range(len(oect_list)):
                globals()['g_list'+str(j)+str(i)].append(oect_list[x][j][i])
            # The initialized data is also recorded, so the length should be +1 
            plt.plot([j for j in range(list_len+1)],globals()['g_list'+str(j)+str(i)],color_list[j],label='weight'+str(j)+str(str(i)))
            plt.legend(loc="upper right")
            plt.xlabel('epochs')
            plt.ylabel('Conductivity/s')
        # /home/chenxi/Documents/python_task/sim_for_oect/weight_visual/
        plt.savefig('weight-column'+str(i)+'.png')
        plt.clf()

def weight_visualize(conductance, padding_len, gif_fps, gif_name_suffix=''):
    '''
    :param conductance: 电导矩阵
    :param padding_len: gif中padding的帧数
    :param gif_fps: gif 帧率
    :param gif_name_suffix: 保存文件名
    :return: 无
    '''

    # 对权重矩阵的序列通过array2gif 进行可视化
    zeros = [np.zeros_like(c) for c in conductance]
    dataset = [np.array([zero, zero, c]) for c, zero in zip(conductance, zeros)]
    padding = np.zeros_like(dataset[0])

    for i in range(padding_len):
        dataset.append(padding)

    normed_dataset = dataset / max([c.max() for c in dataset]) * 255
    # weight_visual/
    write_gif(normed_dataset, '' + gif_name_suffix + '.gif', fps=gif_fps)

