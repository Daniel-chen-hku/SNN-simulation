#!/usr/bin/env python
from data_class import STDP_DATA
from oect_plot import *
import decimal as dec
import numpy as np
import copy
import sys
import os

bais = 1000

def x_dataset():
    signal1 = [1,0,0,1]
    signal2 = [0,1,1,0]
    teacher1 = [0]
    teacher2 = [1]

def check_result(input_signal,stdp_data):
    result = stdp_data.v_input * np.matmul(input_signal, stdp_data.g_oect)
    if float(result) > bais:
        return 1
    return 0 # stand for right

def stdp(in_signal,teacher,stdp_data):
    #input signal
    result = check_result(in_signal,stdp_data)
    stdp_data.delta_t = 100*(np.amax(in_signal) - in_signal - 0.6) #rram + 0.4)
    stdp_data.delta_t = np.round(stdp_data.delta_t,decimals=2)
    if result != teacher:
        if teacher == 0:
            stdp_data.set_reset_flag[0] = -1
        else:
            stdp_data.set_reset_flag[0] = 1
    else:
        stdp_data.flag_clear()
        return 1
    stdp_data.weight_update()
    return 0

def snn_get(testset,answer,stdp_data,type=0):
    error = 0
    for i in range(testset.shape[0]):
        error += check_result(testset[i],answer[i],stdp_data)
    result = dec.Decimal(error*100/testset.shape[0]).quantize(dec.Decimal("0.01"),rounding="ROUND_HALF_UP")
    return result

def snn_learn(trainingset,teacherset,testset,answer,stdp_data):
    assert trainingset.shape[1] == stdp_data.g_oect.shape[0] and teacherset.shape[1] == stdp_data.g_oect.shape[1]
    accuarcy_list = []
    for _epoch in range(stdp_data.epoch):
        for sample in range(trainingset.shape[0]):
            stdp(trainingset[sample],teacherset[sample],stdp_data)
            # if sample%1000 == 0:
            #     print('batch: %2d, acc: %f' % (sample/1000, 1-snn_get(testset,answer,stdp_data)/100))
        error_rate = snn_get(testset,answer,stdp_data)
        print('epoch: %2d, acc: %f' % (_epoch, 1-error_rate/100))
        accuarcy_list.append(copy.deepcopy(1-error_rate/100))
        write_accuracy_log(1-error_rate/100)
        stdp_data.change_learning_rate(beta=0.9)
    draw_accuracy(accuarcy_list)
    return

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    dataset = x_dataset()
    stdp_data = STDP_DATA(learning_rate=1,g_initial=500,v_input = 1,g_size=[4,1],epoch=20,dynamic_rate = 'YES')
    snn_learn(dataset,stdp_data)

