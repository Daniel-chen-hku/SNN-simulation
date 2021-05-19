#!/usr/bin/env python
from array2gif import write_gif
import matplotlib.pyplot as plt
import decimal as dec
import numpy as np
import datetime
import random
import math
import copy
import sys

#copyright:hku eee In-memory calculation group
#author: chenxi
#date:2021/4/13
#this script is suitable for python3.8 or above

#g_list = list() #the current steps of the G change are mA
g_list = [float(dec.Decimal(i).quantize(dec.Decimal("0.01"),rounding="ROUND_HALF_UP")) for i in np.arange(0,100,0.01).tolist()]
g_oect = dict()
delta_i = 0.01 #all current steps are mA
reciprocal_di = 100 #The division of floating-point numbers will bring errors  
delta_t = 0.01 #all time steps are ms
reciprocal_dt = 100
g_initial = 50 #assume the range of variation is [0-100]

class data_set:
    def __init__(self,type=0,setnum=3):
        self.type = type
        self.setnum = setnum
    
    def num_set(self):
        #The training numbers 0, 1, 7 correspond to label 0, 1, 2 respectively
        self.type = 'num'
        input_signal0 = [0.125,0.125,0.125,0.125,0,0.125,0.125,0.125,0.125] #Normalized,stand for 0
        input_signal1 = [0,0.33,0,0,0.33,0,0,0.33,0] #Normalized,stand for 1
        input_signal7 = [0.2,0.2,0.2,0,0,0.2,0,0,0.2] #Normalized,stand for 7
        teacher0 = [1,0,0]
        teacher1 = [0,1,0]
        teacher7 = [0,0,1]
        return (input_signal0,input_signal1,input_signal7,teacher0,teacher1,teacher7)

    def str_set(self):
        #The training str X, J, T correspond to label 0, 1, 2 respectively
        self.type = 'str'
        input_signalX = [1,0,1,0,1,0,1,0,1] #stand for X
        input_signalJ = [0,0,1,0,0,1,1,1,1] #stand for J
        input_signalT = [1,1,1,0,1,0,0,1,0] #stand for T
        teacherX = [1,0,0]
        teacherJ = [0,1,0]
        teacherT = [0,0,1]
        return (input_signalX,input_signalJ,input_signalT,teacherX,teacherJ,teacherT)

    def get_noise_dataset(self,type=0,setnum=5,testnum=10,mode = 1):
        # Parameter Description 
        # setnum: num of dataset; testnum: num of testset; 
        # mode: ways to add noise,1:only black blocks plus noise. 0:all blocks plus noise
        if not type and not self.type:
            print("wrong parameter,no type input")
            return 0
        if 1 != mode and 0 != mode:
            print("wrong parameter,wrong mode")
            return 0
        if setnum < 3 and self.setnum < 3:
            print("wrong parameter,insufficient data length")
            return 0
        if testnum < 5:
            print("wrong parameter,insufficient test data num")
            return 0
        self.setnum = setnum if setnum > self.setnum else self.setnum
        if type:
            self.type = type
        if 'num' == self.type:
            (input_signal0,input_signal1,input_signal2,teacher0,teacher1,teacher2) = self.num_set()
        elif 'str'  == self.type:
            (input_signal0,input_signal1,input_signal2,teacher0,teacher1,teacher2) = self.str_set()
        else:
            print("wrong parameter,woring type input")
            return 0
        dataset = np.zeros((self.setnum*3,9))
        teacherset = np.zeros((self.setnum*3,3))
        testset = np.zeros((testnum,9))
        answer = np.zeros((testnum,3))
        # Mandatory uniform mixing of training data sets 
        a = [0,1,2]
        for i in range(setnum):
            random.shuffle(a)
            for j in a:
                for x in range(len(input_signal0)): 
                    if not mode:
                        dataset[i*3+a.index(j)][x] = (input_signal0,input_signal1,input_signal2)[j][x] + abs(random.gauss(0,0.5))
                    elif (input_signal0,input_signal1,input_signal2)[j][x]:
                        dataset[i*3+a.index(j)][x] = (input_signal0,input_signal1,input_signal2)[j][x] + abs(random.gauss(0,0.5))
                for x in range(len(teacher0)):
                    teacherset[i*3+a.index(j)][x] = (teacher0,teacher1,teacher2)[j][x]
        # create testset and answer
        for i in range(testnum):
            k = random.randint(0,2)
            for j in range(len(input_signal0)):
                if not mode:
                    testset[i][j] = (input_signal0,input_signal1,input_signal2)[k][j] + abs(random.gauss(0,0.5))
                elif (input_signal0,input_signal1,input_signal2)[k][j]:
                    testset[i][j] = (input_signal0,input_signal1,input_signal2)[k][j] + abs(random.gauss(0,0.5))
            for j in range(len(teacher0)):
                answer[i][j] = (teacher0,teacher1,teacher2)[k][j]
        return (dataset,teacherset,testset,answer)

def write_system_log(dataset,teacherset,testset,answer):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = 'snn' + '_simulation' + time_str + 'log.ini'
    sf = open(filename,'w+')
    sf.write('learning sample:\n')
    for i in range(dataset.shape[0]):
        sf.write(str(dataset[i]))
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

def draw_line_chart(oect_list):
    color_list = ['r','y','g','c','b','m','k','teal','skyblue']
    for i in range(3):
        for j in range(9):
            globals()['g_list'+str(j)+str(i)] = list()
            for x in range(len(oect_list)):
                globals()['g_list'+str(j)+str(i)].append(oect_list[x][j][i])
            plt.plot([j for j in range(11)],globals()['g_list'+str(j)+str(i)],color_list[j],label='weight'+str(j)+str(str(i)))
            plt.legend(loc="upper right")
        plt.savefig('/home/chenxi/Documents/python_task/sim_for_oect/weight_visual/weight-column'+str(str(i))+'.png')
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

    write_gif(normed_dataset, 'weight_visual/' + gif_name_suffix + '.gif', fps=gif_fps)

def update_weight_to_show(oect_numpy):
    for i in range(3):
        for x in range(9):
            oect_numpy[x][i] = g_oect["g" + str(x) + str(i)]

def I_integral(v_axon,t_on,t_off,v_ctrl=0.7):
    #After add set or reset signal,calculating the integrated current value
    return 10*v_ctrl*sum(v_axon[int(t_on+0.5):int(t_off+0.5)])*delta_t

def add(t_begin,v_axon,signal='reset'):
    if 'set' != signal and 'reset' != signal:
        print('wrong input of tran ctrl signal')
    elif 'set' == signal:
        return I_integral(v_axon,t_begin,t_begin+0.1*reciprocal_dt)
    elif 'reset' == signal:
        return -I_integral(v_axon,t_begin,t_begin+0.1*reciprocal_dt)

def Axon(t_begin,t_end,v_gate=0.5):
    v_threshold = 0.01
    #v_function = v_gate * exp(-t)
    #ended at v_function < v_threshold
    #t_off = use v_gate and exp(-x) to cal
    t_off = -math.exp(v_threshold/v_gate)
    t = [float(dec.Decimal(i).quantize(dec.Decimal("0.01"),rounding="ROUND_HALF_UP")) for i in np.arange(t_end,t_end + t_off,delta_t).tolist()]
    #print(t_begin,t_end)
    v_axon = int(t_begin+0.5)*[0] + int((t_end - t_begin)+0.5)*[v_gate] + [v_gate*math.exp(-x) for x in t]#v_gate*exp(-x) part
    return v_axon

def G_change(i,g_begin):
    #if reach the two point, warning and stop update the conductivity
    if (-1 == np.sign(i) and g_begin <= g_list[0]) or (1 == np.sign(i) and g_begin >= g_list[-1]):
        print("this is a warning log,the conductivity reach the end point:",g_begin)
        return g_begin
    for x in range(len(g_list)):
        #update the conductivity
        if g_list[x] <= g_begin <= g_list[x+1]:
            if 0 <= x+i*reciprocal_di <= len(g_list):
                return g_list[x+int(i*reciprocal_di+0.5)]
            elif i < 0:
                print("this is a warning log,the conductivity reach the min point")
                return g_list[0]
            else:
                print("this is a warning log,the conductivity reach the max point")
                return g_list[-1]

def stdp(in_signal,teacher):
    global g_oect
    #create signal and input
    for i in range(9):
        globals()["v" + str(i)] = int(in_signal[i]*reciprocal_dt+0.5)*[0.5] + int((max(in_signal) + 1 - in_signal[i])*reciprocal_dt+0.5)*[0]
    #input signal
    result = [0,0,0]
    v_axon = dict()
    for x in range(9):
        #calculate current
        for i in range(3):
            result[i] += sum(globals()["v" + str(x)])*g_oect['g'+str(x)+str(i)]*delta_t
        #add post signal,the total time length is 0.1s
        v_axon[x] = Axon(globals()["v" + str(x)].index(0),globals()["v" + str(x)].index(0)+0.1*reciprocal_dt)
    #the time of when to add set/reset signal
    t_tran = int(max(in_signal)*reciprocal_dt+0.5)
    #compare teach signal,and add set or reset signal
    if result.count(max(result)) != 1:
        #now we only have three output
        #change the conductivity of oect
        for i in range(9):
            g_oect["g" + str(i) + str(teacher.index(1))] = G_change(add(t_tran,v_axon[i],signal='set'),g_oect["g" + str(i) + str(teacher.index(1))])
        #use the first algorithm,change the conductance value of a certain column of oect if its output value is sorted incorrectly
        #if the signal who should be won at the end meanwhile is one of the max value
        if result.count(max(result)) == 3 or result[teacher.index(1)] != max(result):
            for x in range(3):
                if x != teacher.index(1):
                    for i in range(9):
                        g_oect["g" + str(i) + str(x)] = G_change(add(t_tran,v_axon[i]),g_oect["g" + str(i) + str(x)])
        else:
            change_column = result.index(max(result)) if teacher.index(1) != result.index(max(result)) else result.index(max(result),result.index(max(result))+1)
            for i in range(9):
                g_oect["g" + str(i) + str(change_column)] = G_change(add(t_tran,v_axon[i]),g_oect["g" + str(i) + str(change_column)])
    elif result.index(max(result)) != teacher.index(1):
        for i in range(9):
            g_oect["g" + str(i) + str(teacher.index(1))] = G_change(add(t_tran,v_axon[i],signal='set'),g_oect["g" + str(i) + str(teacher.index(1))])
        #If the fire signal is the minimum value, then the three signals must be added with set/reset signals
        if teacher.index(1) == result.index(min(result)):
            for x in range(3):
                if x != teacher.index(1):
                    for i in range(9):
                        g_oect["g" + str(i) + str(x)] = G_change(add(t_tran,v_axon[i]),g_oect["g" + str(i) + str(x)])
        #If the fire signal is the mid value, then only the max value should add the reset signal
        else:
            for i in range(9):
                g_oect["g" + str(i) + str(result.index(max(result)))] = G_change(add(t_tran,v_axon[i]),g_oect["g" + str(i) + str(result.index(max(result)))])
    else:
        return 1
    return 0

def snn_learn_test():
    for i in range(3*9):
        g_oect["g" + str(i//3) + str(i%3)] = g_initial + random.gauss(0,0.1)
    #input data set and teacher signal
    #XJT
    oect_list =list()
    oect_numpy = 50*np.ones([9,3])
    oect_list.append(copy.deepcopy(oect_numpy))
    for i in range(10):
        teacher = [1,0,0]
        input_signal = [1,1,1,1,0,1,1,1,1]#stand for 0
        #training_times = 0
        stdp(input_signal,teacher)
        teacher = [0,1,0]
        input_signal = [0,1,0,0,1,0,0,1,0]#stand for 1
        stdp(input_signal,teacher)
        teacher = [0,0,1]
        input_signal = [1,1,1,0,0,1,0,0,1]#stand for 7
        stdp(input_signal,teacher)
        #update weight to oect_numpy
        update_weight_to_show(oect_numpy)
        oect_list.append(copy.deepcopy(oect_numpy))

def snn_learn(dataset,teacherset):
    if 9 != dataset.shape[1] or 3 != teacherset.shape[1]:
        print("wrong input of snn_learn, input length error")
        return 0
    global g_oect
    #initial the g_list
    #initial the g for each oect
    for i in range(3*9):
        g_oect["g" + str(i//3) + str(i%3)] = g_initial + random.gauss(0,0.1)
    #input data set and teacher signal
    #XJT
    oect_list =list()
    oect_numpy = 50*np.ones([9,3])
    oect_list.append(copy.deepcopy(oect_numpy))
    for i in range(dataset.shape[0]):
        stdp(dataset[i].tolist(),teacherset[i].tolist())
        update_weight_to_show(oect_numpy)
        oect_list.append(copy.deepcopy(oect_numpy))
    # weight_visualize(oect_list,2,1,gif_name_suffix='oect_weight_changes')
    # draw_line_chart(oect_list)
    print(oect_list)
    return 1

def check_result(input_signal,test_result):
    for i in range(9):
        globals()["v" + str(i)] = int(input_signal[i]*reciprocal_dt+0.5)*[0.5] + int((max(input_signal) - input_signal[i])*reciprocal_dt+0.5)*[0]
    #input signal
    result = [0,0,0]
    for x in range(9):
        #calculate current
        for i in range(3):
            result[i] += sum(globals()["v" + str(x)])*g_oect['g'+str(x)+str(i)]*delta_t
    if result.index(max(result)) != test_result.index(1):
        return 1 # stand for wrong output
    return 0 # stand for right

def snn_get_test():
    input_signal = [1,1,1,1,0,1,1,1,1]
    test_result = [1,0,0]
    output = 'right' if not check_result(input_signal,test_result) else 'wrong'
    print(output)
    input_signal = [0,1,0,0,1,0,0,1,0]
    test_result = [0,1,0]
    output = 'right' if not check_result(input_signal,test_result) else 'wrong'
    print(output)
    input_signal = [1,1,1,0,0,1,0,0,1]
    test_result = [0,0,1]
    output = 'right' if not check_result(input_signal,test_result) else 'wrong'
    print(output)
    print(g_oect)

def snn_get(testset,answer):
    error = 0
    for i in range(testset.shape[0]):
        input_signal = testset[i].tolist()
        test_result = answer[i].tolist()
        error += check_result(input_signal,test_result)
    print('error rate:',dec.Decimal(error/testset.shape[0]).quantize(dec.Decimal("0.01"),rounding="ROUND_HALF_UP"))

if __name__ == '__main__':
    dataset = data_set()
    (dataset,teacherset,testset,answer) = dataset.get_noise_dataset(type='str',setnum=50,testnum=20,mode=0)
    # print(dataset)
    # print(teacherset)
    # write_system_log(dataset,teacherset,testset,answer)
    snn_learn(dataset,teacherset)
    snn_get(testset,answer)
    # snn_learn_test()
    # snn_get_test()
