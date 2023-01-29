#!/usr/bin/env python
from data_class import data_set,STDP_DATA
from import_set import load_mnist,load_emnist,set_working_path,write_system_log
from oect_plot import *
import decimal as dec
import numpy as np
import copy
import sys
import os

# copyright:hku eee In-memory computing group
# author:chenxi
# date:2021/4/13
# last modified data:2022/6/15
# this script is suitable for python3.8 or above
# the input signal unit is 100ms 
# usage:pyhon simulation_for_oect.py mnist/emnist/default create/load(only take effect when the previous item is the default)
# test delta t [-60,40] and [-60,60]
# test dynamic learning rate,begin from height,then is the length

def stdp(in_signal,teacher,stdp_data):
    #input signal
    result = stdp_data.v_input * np.matmul(in_signal, stdp_data.g_oect)
    stdp_data.delta_t = 100*(np.amax(in_signal) - in_signal - 0.6) #rram + 0.4)
    stdp_data.delta_t = np.round(stdp_data.delta_t,decimals=2)
    idx_winner_exp = np.argmax(teacher)
    idx_winner_label = np.squeeze( np.argwhere(result == np.amax(result)) )
    # stdp_data.set_reset_flag = np.zeros(3, dtype = int)
    # Winner take all
    # In case of multiple winners, or exp winner is not the supposed winner,
    # SET the supposed winner synapses, RESET the actual winner synapses
    if idx_winner_label.size > 1 or idx_winner_label != idx_winner_exp:
        stdp_data.set_reset_flag[idx_winner_exp] = 1
        # RESET those won but not supposed to win
        idx_to_reset = idx_winner_label[idx_winner_label != idx_winner_exp]
        stdp_data.set_reset_flag[idx_to_reset] = -1
        # reset those who is bigger than the label supposed to win
        # This operation applies to numpy.array, not to list 
        # stdp_data.set_reset_flag[result > result[idx_winner_exp]] = -1
    else:
        stdp_data.flag_clear()
        return 1
    stdp_data.weight_update()
    return 0

def check_result(input_signal,test_result,stdp_data):
    result = stdp_data.v_input * np.matmul(input_signal, stdp_data.g_oect)
    if np.argmax(result) != np.argmax(test_result):
        return 1 # stand for wrong output
    return 0 # stand for right

def get_confusion_matrix(trainingset,teacherset,testset,answer,stdp_data):
    confusion_matrix = np.zeros((5,5))
    # for i in range(trainingset.shape[0]):
    #     result = stdp_data.v_input * np.matmul(trainingset[i], stdp_data.g_oect)
    #     confusion_matrix[np.argmax(teacherset[i])][np.argmax(result)] += 1
    for i in range(testset.shape[0]):
        result = stdp_data.v_input * np.matmul(testset[i], stdp_data.g_oect)
        confusion_matrix[np.argmax(answer[i])][np.argmax(result)] += 1
    confusion_matrix = 5*confusion_matrix / (trainingset.shape[0] + testset.shape[0])
    return confusion_matrix

def snn_get(testset,answer,stdp_data,type=0):
    error = 0
    for i in range(testset.shape[0]):
        error += check_result(testset[i],answer[i],stdp_data)
    result = dec.Decimal(error*100/testset.shape[0]).quantize(dec.Decimal("0.01"),rounding="ROUND_HALF_UP")
    # print('error rate:',result,'%')
    # if type == 1:
    #     write_accuracy_log(result)
    # plot_confusion_matrix(get_confusion_matrix(trainingset,teacherset,testset,answer,stdp_data))
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

# def snn_learn(trainingset,teacherset,testset,answer,stdp_data):
    # assert trainingset.shape[1] == stdp_data.g_oect.shape[0] and teacherset.shape[1] == stdp_data.g_oect.shape[1]
    # # if testset.shape[0] > 1000:
    # #     testset,answer = testset[:1000],answer[:1000]
    # accuarcy_list = list()
    # oect_list = list()
    # # oect_list.append(copy.deepcopy(stdp_data.g_oect))
    # # visualize_list = list()
    # # visualize_list.append(copy.deepcopy(stdp_data.g_oect))
    # for i in range(trainingset.shape[0]):
    #     stdp(trainingset[i],teacherset[i],stdp_data)
    #     accuarcy_list.append(copy.deepcopy(snn_get(testset,answer,stdp_data)))
        # if 0 == (i % 100):
        #     accuarcy_list.append(copy.deepcopy(snn_get(testset,answer,stdp_data)))
        # if 0 == ((i+1) % 100):
        #     visualize_list.append(copy.deepcopy(stdp_data.g_oect))
        # if 0 == ((i+1) % 5000):
        #     oect_list.append(copy.deepcopy(stdp_data.g_oect))
        # oect_list.append(copy.deepcopy(stdp_data.g_oect))
        # accuarcy_list.append(copy.deepcopy(snn_get(testset,answer,stdp_data)))
    # oect_list.append(copy.deepcopy(stdp_data.g_oect)) 
    # weight_visualize(visualize_list,2,1,gif_name_suffix='oect_weight_changes')
    # draw_line_chart(oect_list,list_len=trainingset.shape[0])
    # plot_weight(oect_list)
    # shape_plot_weight(oect_list)
    # write_weight(oect_list)
    # draw_accuracy(accuarcy_list)
    # print(oect_list)
    return 1

def snn_get_test(stdp_data):
    input_signal = np.array([1,1,1,1,0,1,1,1,1])
    test_result = np.array([1,0,0])
    output = 'right' if not check_result(input_signal,test_result,stdp_data) else 'wrong'
    print(output)
    input_signal = np.array([0,1,0,0,1,0,0,1,0])
    test_result = np.array([0,1,0])
    output = 'right' if not check_result(input_signal,test_result,stdp_data) else 'wrong'
    print(output)
    input_signal = np.array([1,1,1,0,0,1,0,0,1])
    test_result = np.array([0,0,1])
    output = 'right' if not check_result(input_signal,test_result,stdp_data) else 'wrong'
    print(output)
    print(stdp_data.g_oect)

def snn_learn_test(stdp_data):
    #input data set and teacher signal
    #XJT
    oect_list =list()
    oect_list.append(copy.deepcopy(stdp_data.g_oect))
    for i in range(10):
        teacher = np.array([1,0,0])
        input_signal = np.array([1,1,1,1,0,1,1,1,1])#stand for 0
        #training_times = 0
        stdp(input_signal,teacher,stdp_data)
        teacher = np.array([0,1,0])
        input_signal = np.array([0,1,0,0,1,0,0,1,0])#stand for 1
        stdp(input_signal,teacher,stdp_data)
        teacher = np.array([0,0,1])
        input_signal = np.array([1,1,1,0,0,1,0,0,1])#stand for 7
        stdp(input_signal,teacher,stdp_data)
        #update weight to oect_numpy
        oect_list.append(copy.deepcopy(stdp_data.g_oect))

if __name__ == '__main__':
    # os.chdir(os.getcwd())
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # data = 'default' if len(sys.argv) <= 1 else str(sys.argv[1])
    data = 'mnist'
    assert data == 'default' or data == 'mnist' or data == 'emnist'
    if 'mnist' == data:
        (trainingset,teacherset,testset,answer) = load_mnist()
    elif 'emnist' == data:
        (trainingset,teacherset,testset,answer) = load_emnist()
    else:
        datasource = 'create' if len(sys.argv) <= 2 else str(sys.argv[2])
        assert datasource == 'create' or datasource == 'load'
        if datasource == 'load':
            dataset = np.load('dataset.npz')
            trainingset,teacherset = dataset['train_data'],dataset['train_label']
            testset,answer = dataset['test_data'],dataset['test_label']
        else:
            dataset = data_set(noise=0.8)
            (trainingset,teacherset,testset,answer) = dataset.get_noise_dataset(type='str',setnum=100,testnum=50,mode=0)
            np.savez('dataset.npz', train_data = trainingset, train_label = teacherset, test_data = testset, test_label = answer)
        draw_sample(trainingset[0].reshape(3,3),trainingset[1].reshape(3,3),trainingset[2].reshape(3,3))
        # write_system_log(trainingset,teacherset,testset,answer)
        # print(trainingset[0],trainingset[1],trainingset[2])
        # snn_learn_test(stdp_data)
        # snn_get_test(stdp_data)
    set_working_path()
    stdp_data = STDP_DATA(learning_rate=1,g_initial=400,g_size=[trainingset.shape[1],teacherset.shape[1]],epoch=20,dynamic_rate = 'YES')
    snn_learn(trainingset,teacherset,testset,answer,stdp_data)
    print('final result:',snn_get(testset,answer,stdp_data))
