#!/usr/bin/env python
from data_class import *
from oect_plot import *
import decimal as dec
import numpy as np
import copy
import sys

# copyright:hku eee In-memory calculation group
# author:chenxi
# date:2021/4/13
# last modified data:2021/7/10
# this script is suitable for python3.8 or above
# the input signal unit is 100ms 
# usage:pyhon simulation_for_oect.py mnist/emnist/default create/load(only take effect when the previous item is the default)

def stdp(in_signal,teacher,stdp_data):
    #input signal
    result = stdp_data.v_input * np.matmul(in_signal, stdp_data.g_oect)
    stdp_data.delta_t = 100*(np.amax(in_signal) - in_signal)
    idx_winner_exp = np.argmax(teacher)
    idx_winner_label = np.squeeze( np.argwhere(result == np.amax(result)) )
    # stdp_data.set_reset_flag = np.zeros(3, dtype = int)
    # Winner take all
    # In case of multiple winners, or exp winner is not the supposed winner,
    # SET the supposed winner synapses, RESET the actual winner synapses
    if idx_winner_label.size > 1 or idx_winner_label != idx_winner_exp:
        stdp_data.set_reset_flag[idx_winner_exp] = 1
        # RESET those won but not supposed to win        
        # idx_to_reset = idx_winner_label[idx_winner_label != idx_winner_exp]
        # stdp_data.set_reset_flag[idx_to_reset] = -1
        # reset those who is bigger than the label supposed to win
        # This operation applies to numpy.array, not to list 
        stdp_data.set_reset_flag[result > result[idx_winner_exp]] = -1
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

def snn_get(testset,answer,stdp_data):
    error = 0
    for i in range(testset.shape[0]):
        error += check_result(testset[i],answer[i],stdp_data)
    result = dec.Decimal(error*100/testset.shape[0]).quantize(dec.Decimal("0.01"),rounding="ROUND_HALF_UP")
    print('error rate:',result,'%')
    write_accuracy_log(result)
    return result

def snn_learn(trainingset,teacherset,testset,answer,stdp_data):
    assert trainingset.shape[1] == stdp_data.g_oect.shape[0] and teacherset.shape[1] == stdp_data.g_oect.shape[1]
    accuarcy_list = list()
    #input data set and teacher signal
    #XJT
    oect_list =list()
    oect_list.append(copy.deepcopy(stdp_data.g_oect))
    for i in range(trainingset.shape[0]):
        stdp(trainingset[i],teacherset[i],stdp_data)
        oect_list.append(copy.deepcopy(stdp_data.g_oect))
        accuarcy_list.append(copy.deepcopy(snn_get(testset,answer,stdp_data)))
    weight_visualize(oect_list,2,20,gif_name_suffix='oect_weight_changes')
    draw_line_chart(oect_list,list_len=trainingset.shape[0])
    draw_accuracy(accuarcy_list)
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
    os.chdir(os.path.dirname(__file__))
    datasource = 'create' if len(sys.argv) <= 1 else str(sys.argv[1])
    assert datasource == 'create' or datasource == 'load'
    if datasource == 'load':
        dataset = np.load('dataset.npz')
        trainingset = dataset['train_data']
        teacherset = dataset['train_label']
        testset = dataset['test_data']
        answer = dataset['test_label']
    else:
        dataset = data_set(noise=0.4)
        (trainingset,teacherset,testset,answer) = dataset.get_noise_dataset(type='str',setnum=100,testnum=60,mode=0)
        np.savez('dataset.npz', train_data = trainingset, train_label = teacherset, test_data = testset, test_label = answer)
    set_working_path()
    stdp_data = STDP_DATA(learning_rate=0.1,g_initial=200,g_size=[9,3])
    # write_system_log(trainingset,teacherset,testset,answer)
    draw_sample(trainingset[0].reshape(3,3),trainingset[1].reshape(3,3),trainingset[2].reshape(3,3))
    # print(trainingset[0],trainingset[1],trainingset[2])
    snn_learn(trainingset,teacherset,testset,answer,stdp_data)
    snn_get(testset,answer,stdp_data)
    # snn_learn_test(stdp_data)
    # snn_get_test(stdp_data)
    dataset.close()
