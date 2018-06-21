# -*- coding: utf-8 -*-
##
##  tfcxxtxx.py
##  refer ...
##  arranged by y.m. 2017/9/13-
##  rel 01
##  py tfcxx.py 
## 		1 input-file ex.USDJPY1440.csv
##		2 type    Open/High/Low/Close
##		3 step    3/21
##		4 future 1(deault)
##		5 ratio  0.01 x sratio default...10/50
##		6 layer mode 1...normal unit=16/2...3layers
##		7 pedict 1/2/3/4/5...
##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tflearn
import sys


from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

csvfname =  sys.argv[1]
cltype   =  sys.argv[2]
sstep    =  sys.argv[3]
sfuture  =  sys.argv[4]
sratio   =  sys.argv[5]
slayer   =  sys.argv[6]
spredict  =  sys.argv[7]



class Prediction :

    def __init__(self):
        self.dataset = None
        # initial data
        self.model = None
        self.train_predict = None
        self.test_predict = None
        self.curmin = 0.0
        self.curmax = 0.0
        self.csvfn = None
        self.csvcl = None

        # set data
        self.steps_of_history = int(sstep)
        self.steps_in_future = int(sfuture)
        self.csvfn =  csvfname
        self.csvcl =  cltype
        print( csvfname )

    def load_dataset(self):
        # prepare data
        dataframe = pd.read_csv(self.csvfn,
                usecols=([cltype]),
                engine='python')
        self.dataset = dataframe.values
        self.dataset = self.dataset.astype('float32')

        # normalize
        self.curmin = np.min(np.abs(self.dataset))
        self.dataset -= self.curmin
        self.curmax = np.max(np.abs(self.dataset))
        self.dataset /= self.curmax
        print("MINMAX")
        print(self.curmin)
        print(self.curmax)


    def create_dataset(self):
        X, Y = [], []
        for i in range(0, len(self.dataset) - self.steps_of_history, self.steps_in_future):
            X.append(self.dataset[i:i + self.steps_of_history])
            if (i + self.steps_of_history+int(spredict)) < len(self.dataset):
                Y.append(self.dataset[i + self.steps_of_history+int(spredict)])

        X = np.reshape(np.array(X), [-1, self.steps_of_history, 1])
        Y = np.reshape(np.array(Y), [-1, 1])

        print("Xre" )
        print( X )
        print( "Yre" )
        print( Y )
        return X, Y

    def setup(self):
        self.load_dataset()
        X, Y = self.create_dataset()

        # Build neural network
        #1...Hidden Layer:1/Units =16
        #    activation:tahh/linear/softmax/relu
        if slayer == str("1"):
            net = tflearn.input_data(shape=[None, self.steps_of_history, 1])
            net = tflearn.lstm(net, n_units=16)
            net = tflearn.fully_connected(net, 1, activation='linear')
        #
        elif slayer == str("2"):
            net = tflearn.input_data(shape=[None, self.steps_of_history, 1],name="input_layer")
            net = tflearn.lstm(net, n_units=100, activation='tanh', dropout=0.9, return_seq=True, return_state=False,name="hidden_layer_1")
            net = tflearn.lstm(net, n_units=50, activation='tanh', dropout=0.9, return_seq=True, return_state=False,name="hidden_layer_2")
            net = tflearn.lstm(net, n_units=25, activation='tanh', dropout=0.9, name="hidden_layer_3")
            net = tflearn.fully_connected(net, 1, activation='tanh',name="output_layer")

        # method
        #    optimizer:adam/sgd
        #    loss:mean_square/categorical_...
        net = tflearn.regression(net, optimizer='adam', learning_rate=0.0005,
                loss='mean_square')

        # Define model
        self.model = tflearn.DNN(net, tensorboard_verbose=0)

        # input data ratio
        pos = round(len(X) * (1 - 0.01*int(sratio) ))
        trainX, trainY = X[:pos], Y[:pos]
        testX, testY   = X[pos:len(X)+int(spredict)], Y[pos:len(Y)+int(spredict)]

        return trainX, trainY, testX

    def executePredict(self, trainX, trainY, testX):
        # Start training (apply gradient descent algorithm)
        self.model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=1, n_epoch=150, run_id='Currency')

        # predict
        self.train_predict = self.model.predict(trainX)
        self.test_predict = self.model.predict(testX)

    def showResult(self):
        # plot train data
        data_s, test_s = [], []
        train_predict_plot = np.empty_like(self.dataset)
        train_predict_plot[:, :] = np.nan
        train_predict_plot[self.steps_of_history:len(self.train_predict) + self.steps_of_history, :] = \
                self.train_predict*self.curmax+self.curmin

        # plot test dat
        test_predict_plot = np.empty_like(self.dataset)
        test_predict_plot[:, :] = np.nan
        test_predict_plot[len(self.train_predict) + self.steps_of_history:len(self.dataset), :] = \
                self.test_predict*self.curmax+self.curmin

        for i in range(0, len(self.dataset)):
            self.dataset[i]=self.dataset[i]*self.curmax+self.curmin
        print("SD" )
        print(self.dataset[0:len(self.dataset)])
        #print(len(self.dataset))
        #print(len(train_predict_plot))
        #print(train_predict_plot[0:len(train_predict_plot)])
        #print(len(self.test_predict))
        #print(self.test_predict[0:len(self.test_predict)])

        print("RESULT")
        for i in range(0, len(self.test_predict)):
            self.test_predict[i]=self.test_predict[i]*self.curmax+self.curmin
        #

        print("Test Predict" )
        print(self.test_predict[len(self.test_predict)-10:len(self.test_predict)])
        
        # plot show res
        plt.figure(figsize=(16, 16))
        plt.title('History={} Future={}'.format(self.steps_of_history, self.steps_in_future)+self.csvfn)
        plt.plot(self.dataset[len(self.dataset)-len(self.test_predict)-int(spredict):len(self.dataset)], label="actual", color="k")
        #plt.plot(train_predict_plot[,:], label="train", color="r")
        plt.plot(self.test_predict[:len(self.test_predict)], label="test", color="b")
        plt.savefig('B'+self.csvfn+'result.png')
        #plt.show()

        plt.figure(figsize=(16, 16))
        plt.title('History={} Future={}'.format(self.steps_of_history, self.steps_in_future))
        plt.plot(self.dataset, label="actual", color="k")
        plt.plot(train_predict_plot, label="train", color="r")
        plt.plot(test_predict_plot, label="test", color="b")
        plt.savefig('S'+self.csvfn+'result.png')

if __name__ == "__main__":

    print( "Input:"+ csvfname )
    print( "Type :"+ cltype   )
    print( "Step :"+ sstep    )
    print( "Future:" +sfuture )
    print( "Ratio:"  + sratio )
    print( "Layer mode:" + slayer )
    print( "Predic:"+spredict  )

    prediction = Prediction()
    trainX, trainY, testX = prediction.setup()
    prediction.executePredict(trainX, trainY, testX)
    prediction.showResult()
