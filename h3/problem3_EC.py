# CS 181, Harvard University
# Spring 2016
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c
from Perceptron import Perceptron
import random
import time
from sklearn import svm

# Implement this class
class SMO(Perceptron):
    def __init__(self, numsamples):
        self.numsamples = numsamples

    def __Kernel1(self,X,XP):
        return np.dot(X,XP)

    def __Kernel(self,X,XP):
        return self.__Kernel1(X,XP)

    def __CheckAccuracy(self,X,Y):
        v=self.predict(X)
        print 'Percent correct:',100-100*sum(abs(v-Y))/(2*len(Y))

    def __ImportValidationSet(self):
        data = np.loadtxt("data.csv", delimiter=',')
        X = data[:, :2]
        Y = data[:, 2]
        return X,Y

    def __ComputeGradient(self,X,Y,alpha,k):
        yHat=0
        for i in xrange(len(alpha)):
            yHat+=alpha[i]*self.__Kernel(X[k],X[i])
        g=Y[k]-yHat
        return g

    def __SciKitLearn(self,XTest,YTest,X,Y):
        clf = svm.SVC()
        bt=time.time()
        clf.fit(X,Y)
        print 'SciKit Learn Training Time:', time.time() - bt
        v=clf.predict(XTest)
        print 'SciKit Learn Percent correct:',100-100*sum(abs(v-YTest))/(2*len(YTest))


    # Implement this!
    def fit(self, X, Y):
        print 'Fitting SMO'
        print '--------------------------------------'
        print '--------------------------------------'

        Tau=0.1
        C=5

        breakFlag=True

        bt=time.time()
        numSamples=self.numsamples
        numOfPoints=len(X)

        numOfTrain=int(1./3.*numOfPoints)
        XTest=X[numOfTrain:]
        YTest=Y[numOfTrain:]

        X=X[:numOfTrain]
        Y=Y[:numOfTrain]

        self.X=X
        self.Y=Y

        alpha={x:0 for x in xrange(numOfTrain)}
        A=np.zeros((len(Y),1));B=np.zeros((len(Y),1))
        for i in range(len(Y)):
            A[i]=np.min([0,Y[i]*C])
            B[i]=np.max([0,Y[i]*C])

        g=np.zeros((len(X),1))
        for k in range(len(X)):
            g[k]=self.__ComputeGradient(X, Y, alpha,k)

        while breakFlag:
            breakFlagInside=True
            for i in xrange(len(X)):
                for j in xrange(len(X)):
                    if alpha[i]<B[i] and alpha[j]>A[j] and g[i]-g[j]>Tau:
                        minVec=[(g[i]-g[j])/(self.__Kernel(X[i],X[i])+self.__Kernel(X[j],X[j])+2*self.__Kernel(X[i],X[j]))]
                        minVec.append(B[i]-alpha[i])
                        minVec.append(alpha[i]-A[i])
                        lam=min(minVec)
                        alpha[i]=alpha[i]+lam
                        alpha[j]=alpha[j]-lam
                        for s in range(len(X)):
                            g[s] += -lam*(self.__Kernel(X[i],X[s])-self.__Kernel(X[j],X[s]))
                        breakFlagInside=False
            if breakFlagInside:
                breakFlag=False


        for i in xrange(numOfTrain):
            for j in xrange(len(X[i])):
                w[j]+=alpha[i]*X[i,j]

        self.w=w
        print 'Fitting Kernel Perceptron Done!'
        print 'Training Time:',time.time()-bt
        print 'Number of Support Vectors:',len(S)

        self.__CheckAccuracy(XTest,YTest)
        self.__SciKitLearn(XTest,YTest,X,Y)

        print '--------------------------------------'
        print 'Using Validation Set'
        [XValidation,YValidation]=self.__ImportValidationSet()
        self.__CheckAccuracy(XValidation,YValidation)
        self.__SciKitLearn(XValidation,YValidation,X,Y)
        print '--------------------------------------'


    # Implement this!
    def predict(self, X):
        v=np.sign(np.dot(np.transpose(self.w), np.transpose(X)))
        v[v==0]=1
        return v



# Do not change these three lines.
data = np.loadtxt("data.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.

kernel_file_name = 'SMO.png'

# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.
k = SMO(numsamples)
k.fit(X,Y)
k.visualize(kernel_file_name, width=0, show_charts=False, save_fig=True, include_points=False)