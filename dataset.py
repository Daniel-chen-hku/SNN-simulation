
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

        '''
        # How about something like this
        raw_patterns = np.array([input_signal0, input_signal2, input_signal2])
        i_train = np.array([0, 1, 2])
        i_train = np.repeat(i_train, self.setnum)
        i_train = np.shuffle(i_train)
        raw_dataset = raw_patterns[i_train]
        noise = np.abs(np.random.randn(dataset.shape))
        dataset = raw_dataset + noise
        '''

        for i in range(setnum):
            random.shuffle(a)
            for j in a:
                for x in range(len(input_signal0)):                     
                    if not mode:
                        dataset[i*3+a.index(j)][x] = (input_signal0,input_signal1,input_signal2)[j][x] + abs(random.gauss(0,noise))
                        dataset[i*3+a.index(j)][x] = dec.Decimal(dataset[i*3+a.index(j)][x]).quantize(dec.Decimal("0.01"),rounding="ROUND_HALF_UP")
                    elif (input_signal0,input_signal1,input_signal2)[j][x]:
                        dataset[i*3+a.index(j)][x] = (input_signal0,input_signal1,input_signal2)[j][x] + abs(random.gauss(0,noise))
                        dataset[i*3+a.index(j)][x] = dec.Decimal(dataset[i*3+a.index(j)][x]).quantize(dec.Decimal("0.01"),rounding="ROUND_HALF_UP")
                for x in range(len(teacher0)):
                    teacherset[i*3+a.index(j)][x] = (teacher0,teacher1,teacher2)[j][x]
        # create testset and answer
        for i in range(testnum):
            k = random.randint(0,2)
            for j in range(len(input_signal0)):
                if not mode:
                    testset[i][j] = (input_signal0,input_signal1,input_signal2)[k][j] + abs(random.gauss(0,noise))
                    testset[i][j] = dec.Decimal(testset[i][j]).quantize(dec.Decimal("0.01"),rounding="ROUND_HALF_UP")
                elif (input_signal0,input_signal1,input_signal2)[k][j]:
                    testset[i][j] = (input_signal0,input_signal1,input_signal2)[k][j] + abs(random.gauss(0,noise))
                    testset[i][j] = dec.Decimal(testset[i][j]).quantize(dec.Decimal("0.01"),rounding="ROUND_HALF_UP")
            for j in range(len(teacher0)):
                answer[i][j] = (teacher0,teacher1,teacher2)[k][j]
        return (dataset,teacherset,testset,answer)
