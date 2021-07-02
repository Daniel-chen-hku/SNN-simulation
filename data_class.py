#!/usr/bin/env python
import decimal as dec
import numpy as np
import random
import math

class data_set:
    def __init__(self,type=0,setnum=3,noise=0.1):
        self.type = type
        self.setnum = setnum
        self.noise = noise
    
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
                        dataset[i*3+a.index(j)][x] = (input_signal0,input_signal1,input_signal2)[j][x] + abs(random.gauss(0,self.noise))
                        dataset[i*3+a.index(j)][x] = dec.Decimal(dataset[i*3+a.index(j)][x]).quantize(dec.Decimal("0.01"),rounding="ROUND_HALF_UP")
                    elif (input_signal0,input_signal1,input_signal2)[j][x]:
                        dataset[i*3+a.index(j)][x] = (input_signal0,input_signal1,input_signal2)[j][x] + abs(random.gauss(0,self.noise))
                        dataset[i*3+a.index(j)][x] = dec.Decimal(dataset[i*3+a.index(j)][x]).quantize(dec.Decimal("0.01"),rounding="ROUND_HALF_UP")
                for x in range(len(teacher0)):
                    teacherset[i*3+a.index(j)][x] = (teacher0,teacher1,teacher2)[j][x]
        # create testset and answer
        for i in range(testnum):
            k = random.randint(0,2)
            for j in range(len(input_signal0)):
                if not mode:
                    testset[i][j] = (input_signal0,input_signal1,input_signal2)[k][j] + abs(random.gauss(0,self.noise))
                    testset[i][j] = dec.Decimal(testset[i][j]).quantize(dec.Decimal("0.01"),rounding="ROUND_HALF_UP")
                elif (input_signal0,input_signal1,input_signal2)[k][j]:
                    testset[i][j] = (input_signal0,input_signal1,input_signal2)[k][j] + abs(random.gauss(0,self.noise))
                    testset[i][j] = dec.Decimal(testset[i][j]).quantize(dec.Decimal("0.01"),rounding="ROUND_HALF_UP")
            for j in range(len(teacher0)):
                answer[i][j] = (teacher0,teacher1,teacher2)[k][j]
        return (dataset,teacherset,testset,answer)

class STDP_DATA:
    def __init__(self, g_initial, v_input = 0.5, learning_rate = 0.1, g_size = [9,3], dynamic_rate = 'NO'):
        self.learning_rate = learning_rate # control the length of v_gate
        self.rate_type = dynamic_rate
        self.size = g_size
        self.v_input = v_input
        self.delta_t = [0 for i in range(self.size[0])]
        self.change_flag = [0 for i in range(self.size[1])] # -1 present reset while 1 present set
        self.g_oect = np.ones(self.size)
        for i in np.nditer(self.g_oect, order='C',op_flags=['readwrite']):
            i[...] = random.gauss(g_initial,0.4)

    def depression(self,x):
        # return 1.006e-09 * (x**4) - 7.358e-07 * (x**3) + 0.000173 * (x**2) - 0.01869 * x + 1.173
        return 0.06 - 1.81 * math.exp(-x/65.34)

    def exicitation(self,x):
        # return 
        return 0.24 + 1.43 * math.exp(-x/57.73)
    
    def set(self,i,j):
        assert self.delta_t[j] > -40 
        time_list = [-40,20,60,120,200,400]
        # g_change = [2.24,0.90435,0.46925,0.3907,0.07,0.02465]
        v_list = [0.496,0.11858,0.14,0.16,0.011,0.0539]
        for t in time_list:
            if t < self.delta_t[j]:
                continue
            t0 = time_list[time_list.index(t)-1]
            variance = (t - self.delta_t[j]) / (t - t0) * v_list[time_list.index(t)-1] + (self.delta_t[j] - t0) / (t - t0) * v_list[time_list.index(t)]
            self.g_oect[j][i] += random.gauss(self.exicitation(self.delta_t[j]),variance)
        self.g_oect[j][i] += random.gauss(self.exicitation(self.delta_t[j]),0.0539)

    def reset(self,i,j):
        assert self.delta_t[j] > -40 
        # change weight self.g_oect[j][i]
        time_list = [-40,20,60,120,200,400]
        # g_change = [3.125,1.128,0.81,0.58,0.295,0.1195]
        v_list = [0.222,0.415,0.276,0.355,0.023,0.04517]
        for t in time_list:
            if t < self.delta_t[j]:
                continue
            t0 = time_list[time_list.index(t)-1]
            variance = (t - self.delta_t[j]) / (t - t0) * v_list[time_list.index(t)-1] + (self.delta_t[j] - t0) / (t - t0) * v_list[time_list.index(t)]
            self.g_oect[j][i] += random.gauss(self.depression(self.delta_t[j]),variance)
        self.g_oect[j][i] += random.gauss(self.depression(self.delta_t[j]),0.04517)

    def weight_update(self):
        # assert 0 not in self.change_flag
        for i in range(self.size[1]):
            if 1 == self.change_flag[i]: # can not use is or is not, because the variable is in a list
                for j in range(self.size[0]):
                    self.set(i,j)
            if -1 == self.change_flag[i]:
                for j in range(self.size[0]):
                    self.reset(i,j)
        # reset change_flag after weight updata
        self.delta_t = [0 for i in range(self.size[0])]
        self.change_flag = [0 for i in range(self.size[1])]
    
    def flag_clear(self):
        self.delta_t = [0 for i in range(self.size[0])]
        self.change_flag = [0 for i in range(self.size[1])]