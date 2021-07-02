#!/usr/bin/env python
from oect_plot import *
from data_class import *
import decimal as dec
import numpy as np
import random
import math
import copy
import sys

# copyright:hku eee In-memory calculation group
# author: chenxi
# date:2021/4/13
# last modified data:2021/7/2
# this script is suitable for python3.8 or above
# the input signal unit is 100ms 

def stdp(in_signal,teacher,stdp_data):
    #input signal
    result = [0,0,0]
    for x in range(9):
        #calculate current
        for i in range(3):
            result[i] += stdp_data.v_input * in_signal[x] * stdp_data.g_oect[x][i]
        # calculate the delta t
        stdp_data.delta_t[x] = 100 * ( max(in_signal) - in_signal[x] )
    #compare teach signal,and add set or reset signal
    if result.count(max(result)) != 1:
        #now we only have three output
        #change the conductivity of oect
        stdp_data.change_flag[teacher.index(1)] = 1 # set signal
        #use the first algorithm,change the conductance value of a certain column of oect if its output value is sorted incorrectly
        #if the signal who should be won at the end meanwhile is one of the max value
        if result.count(max(result)) == 3 or result[teacher.index(1)] != max(result):
            for x in range(3):
                if x != teacher.index(1):
                    stdp_data.change_flag[x] = -1 # reset signal
        else:
            change_column = result.index(max(result)) if teacher.index(1) != result.index(max(result)) else result.index(max(result),result.index(max(result))+1)
            stdp_data.change_flag[change_column] = -1
    elif result.index(max(result)) != teacher.index(1):
        stdp_data.change_flag[teacher.index(1)] = 1
        #If the fire signal is the minimum value, then the three signals must be added with set/reset signals
        if teacher.index(1) == result.index(min(result)):
            for x in range(3):
                if x != teacher.index(1):
                    stdp_data.change_flag[x] = -1
        #If the fire signal is the mid value, then only the max value should add the reset signal
        else:
            stdp_data.change_flag[result.index(max(result))] = -1
    else:
        stdp_data.flag_clear()
        return 1
    stdp_data.weight_update()
    return 0

def check_result(input_signal,test_result,stdp_data):
    for i in range(9):
        globals()["v" + str(i)] = input_signal[i]
    #input signal
    result = [0,0,0]
    for x in range(9):
        #calculate current
        for i in range(3):
            result[i] += stdp_data.v_input * globals()["v" + str(x)] * stdp_data.g_oect[x][i]
    if result.index(max(result)) != test_result.index(1):
        return 1 # stand for wrong output
    return 0 # stand for right

def snn_get(testset,answer,stdp_data):
    error = 0
    for i in range(testset.shape[0]):
        input_signal = testset[i].tolist()
        test_result = answer[i].tolist()
        error += check_result(input_signal,test_result,stdp_data)
    result = dec.Decimal(error/testset.shape[0]).quantize(dec.Decimal("0.01"),rounding="ROUND_HALF_UP")
    print('error rate:',result)
    sf = open('Accuracy.log','w+')
    sf.write(str(result))
    sf.write('\n')
    sf.close()
    return result

def snn_learn(dataset,teacherset,testset,answer,stdp_data):
    if 9 != dataset.shape[1] or 3 != teacherset.shape[1]:
        print("wrong input of snn_learn, input length error")
        return 0
    accuarcy_list = list()
    #input data set and teacher signal
    #XJT
    oect_list =list()
    oect_list.append(copy.deepcopy(stdp_data.g_oect))
    for i in range(dataset.shape[0]):
        stdp(dataset[i].tolist(),teacherset[i].tolist(),stdp_data)
        oect_list.append(copy.deepcopy(stdp_data.g_oect))
        accuarcy_list.append(copy.deepcopy(snn_get(testset,answer,stdp_data)))
    weight_visualize(oect_list,2,20,gif_name_suffix='oect_weight_changes')
    draw_line_chart(oect_list,list_len=dataset.shape[0])
    draw_accuracy(accuarcy_list)
    # print(oect_list)
    return 1

def snn_get_test(stdp_data):
    input_signal = [1,1,1,1,0,1,1,1,1]
    test_result = [1,0,0]
    output = 'right' if not check_result(input_signal,test_result,stdp_data) else 'wrong'
    print(output)
    input_signal = [0,1,0,0,1,0,0,1,0]
    test_result = [0,1,0]
    output = 'right' if not check_result(input_signal,test_result,stdp_data) else 'wrong'
    print(output)
    input_signal = [1,1,1,0,0,1,0,0,1]
    test_result = [0,0,1]
    output = 'right' if not check_result(input_signal,test_result,stdp_data) else 'wrong'
    print(output)
    print(stdp_data.g_oect)

def snn_learn_test(stdp_data):
    #input data set and teacher signal
    #XJT
    oect_list =list()
    oect_list.append(copy.deepcopy(stdp_data.g_oect))
    for i in range(10):
        teacher = [1,0,0]
        input_signal = [1,1,1,1,0,1,1,1,1]#stand for 0
        #training_times = 0
        stdp(input_signal,teacher,stdp_data)
        teacher = [0,1,0]
        input_signal = [0,1,0,0,1,0,0,1,0]#stand for 1
        stdp(input_signal,teacher,stdp_data)
        teacher = [0,0,1]
        input_signal = [1,1,1,0,0,1,0,0,1]#stand for 7
        stdp(input_signal,teacher,stdp_data)
        #update weight to oect_numpy
        oect_list.append(copy.deepcopy(stdp_data.g_oect))

if __name__ == '__main__':
    # os.chdir(os.path.dirname(__file__)) 
    dataset = data_set()
    (dataset,teacherset,testset,answer) = dataset.get_noise_dataset(type='str',setnum=100,testnum=50,mode=0)
    stdp_data = STDP_DATA(learning_rate=0.1,g_initial=200,g_size = [9,3])
    write_system_log(dataset,teacherset,testset,answer)
    draw_sample(dataset[0].reshape(3,3),dataset[1].reshape(3,3),dataset[2].reshape(3,3))
    # print(dataset[0],dataset[1],dataset[2])
    snn_learn(dataset,teacherset,testset,answer,stdp_data)
    snn_get(testset,answer,stdp_data)
    # snn_learn_test(stdp_data)
    # snn_get_test(stdp_data)
