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

    def get_noise_dataset(self,type=0,setnum=5,testnum=10,mode = 0):
        # Parameter Description 
        # setnum: num of dataset; testnum: num of testset; 
        # mode: ways to add noise,1:only black blocks plus noise. 0:all blocks plus noise
        # assert  (type or self.type) and (1 == mode or 0 == mode) and (setnum >= 3 or self.setnum >= 3) and testnum > 5
        self.setnum = setnum if setnum > self.setnum else self.setnum
        if type:
            self.type = type
        assert 'num' == self.type or 'str'  == self.type
        if 'num' == self.type:
            (input_signal0,input_signal1,input_signal2,teacher0,teacher1,teacher2) = self.num_set()
        elif 'str'  == self.type:
            (input_signal0,input_signal1,input_signal2,teacher0,teacher1,teacher2) = self.str_set()

        # idx_dataset = np.repeat([0, 1, 2], setnum + testnum)
        # rng = np.random.default_rng()
        # rng.shuffle(idx_dataset)
        # original_data = np.array([input_signal0, input_signal1, input_signal2])
        # original_label = np.array([teacher0, teacher1, teacher2])
        # raw_data_set = original_data[idx_dataset]
        # raw_label_set = original_label[idx_dataset]
        # # Add noise
        # noise = np.random.randn(*raw_data_set.shape) * self.noise
        # # noise_abs = np.abs(np.random.randn(*raw_data_set.shape) * self.noise)
        # if 0 == mode:
        #     raw_data_set = raw_data_set + noise
        #     raw_data_set = np.abs(raw_data_set)
        #     # raw_data_set = raw_data_set + noise
        # else:
        #     raw_data_set[raw_data_set != 0] = raw_data_set[raw_data_set != 0] + noise[raw_data_set != 0]
        # # Divide to train/test        
        # idx_train = idx_dataset[0 : setnum * 3]
        # idx_test = idx_dataset[- testnum * 3:]
        # train_data_set = raw_data_set[idx_train]
        # train_data_set = np.round(train_data_set,decimals=2)
        # test_data_set = raw_data_set[idx_test]
        # test_data_set = np.round(test_data_set,decimals=2)
        # train_label_set = raw_label_set[idx_train]
        # test_label_set = raw_label_set[idx_test]
        # return (train_data_set, train_label_set, test_data_set, test_label_set)
        trainingset = np.zeros((self.setnum*3,9))
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
                        trainingset[i*3+a.index(j)][x] = (input_signal0,input_signal1,input_signal2)[j][x] + abs(random.gauss(0,self.noise))
                        trainingset[i*3+a.index(j)][x] = dec.Decimal(trainingset[i*3+a.index(j)][x]).quantize(dec.Decimal("0.01"),rounding="ROUND_HALF_UP")
                    elif (input_signal0,input_signal1,input_signal2)[j][x]:
                        trainingset[i*3+a.index(j)][x] = (input_signal0,input_signal1,input_signal2)[j][x] + abs(random.gauss(0,self.noise))
                        trainingset[i*3+a.index(j)][x] = dec.Decimal(trainingset[i*3+a.index(j)][x]).quantize(dec.Decimal("0.01"),rounding="ROUND_HALF_UP")
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
        return (trainingset,teacherset,testset,answer)

    def close(self):
        return

class STDP_DATA:
    def __init__(self, g_initial, v_input = 0.5, learning_rate = 0.1, g_size = [9,3], dynamic_rate = 'NO'):
        self.learning_rate = learning_rate # control the length of v_gate
        self.rate_type = dynamic_rate
        self.size = g_size
        self.v_input = v_input
        self.delta_t = np.zeros(self.size[0],dtype = int)
        self.set_reset_flag = np.zeros(self.size[1],dtype = int) # -1 present reset while 1 present set
        self.g_oect = g_initial + np.random.randn(*self.size)
        self.g_oect[self.g_oect < 0] = 0
        self.time_set_list = [-40,20,60,120,200,400]
        self.v_set_list = [0.496,0.11858,0.14,0.16,0.011,0.0539]
        self.time_reset_list = [-40,20,60,120,200,400]        
        self.v_reset_list = [0.222,0.415,0.276,0.355,0.023,0.04517]

    def depression(self,x):
        # return 1.006e-09 * (x**4) - 7.358e-07 * (x**3) + 0.000173 * (x**2) - 0.01869 * x + 1.173
        return 0.06 - 1.81 * math.exp(-x/65.34)

    def exicitation(self,x):
        # return 
        return 0.24 + 1.43 * math.exp(-x/57.73)
    
    def set_reset(self,i,j, is_set=True):
        assert self.delta_t[j] > -40
        if is_set:
            time_list = self.time_set_list
            v_list = self.v_set_list
        else:
            time_list = self.time_reset_list
            v_list = self.v_reset_list

        delta_g = self.exicitation(self.delta_t[j]) if is_set else self.depression(self.delta_t[j])

        for t in time_list:
            if t < self.delta_t[j]:
                continue
            t0 = time_list[time_list.index(t)-1]
            variance = (t - self.delta_t[j]) / (t - t0) * v_list[time_list.index(t)-1] + (self.delta_t[j] - t0) / (t - t0) * v_list[time_list.index(t)]
            self.g_oect[j][i] += random.gauss(delta_g,variance)
            return 0
        self.g_oect[j][i] += random.gauss(delta_g, v_list[-1])
        return 1

    def weight_update(self):
        # assert 0 not in self.set_reset_flag
        for i in range(self.size[1]):
            if 1 == self.set_reset_flag[i]: # can not use is or is not, because the variable is in a list
                for j in range(self.size[0]):
                    self.set_reset(i,j,is_set=True)
            if -1 == self.set_reset_flag[i]:
                for j in range(self.size[0]):
                    self.set_reset(i,j,is_set=False)
        # reset set_reset_flag after weight updata
        self.delta_t = np.zeros(self.size[0],dtype = int)
        self.set_reset_flag = np.zeros(self.size[1],dtype = int)
    
    def flag_clear(self):
        self.delta_t = np.zeros(self.size[0],dtype = int)
        self.set_reset_flag = np.zeros(self.size[1],dtype = int)