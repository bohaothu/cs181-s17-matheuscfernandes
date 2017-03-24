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
class KernelPerceptron(Perceptron):
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

    def __SciKitLearn(self,XTest,YTest,X,Y):
        clf = svm.SVC()
        bt=time.time()
        clf.fit(X,Y)
        print 'SciKit Learn Training Time:', time.time() - bt
        v=clf.predict(XTest)
        print 'SciKit Learn Percent correct:',100-100*sum(abs(v-YTest))/(2*len(YTest))


    # Implement this!
    def fit(self, X, Y):
        print 'Fitting Kernel Perceptron'
        print '--------------------------------------'
        print '--------------------------------------'

        bt=time.time()
        numSamples=self.numsamples
        numOfPoints=len(X)

        numOfTrain=int(2./3.*numOfPoints)
        XTest=X[numOfTrain:]
        YTest=Y[numOfTrain:]

        X=X[:numOfTrain]
        Y=Y[:numOfTrain]

        self.X=X
        self.Y=Y

        S=[]; alpha={x:0 for x in xrange(numOfTrain)}

        for num in xrange(numSamples):

            t=random.randint(0,numOfTrain-1)
            Xt=X[t];Yt=Y[t]
            YHatT=0

            for i in S:
                YHatT+=alpha[i]*self.__Kernel(Xt,X[i])

            if Yt*YHatT <=0:
                S.append(t)
                alpha[t]=Yt

        w=[0,0]

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


# Implement this class
class BudgetKernelPerceptron(Perceptron):
    def __init__(self, beta, N, numsamples):
        self.beta = beta
        self.N = N
        self.numsamples = numsamples

    def __Kernel1(self,X,XP):
        return np.dot(X,XP)

    def __Kernel(self,X,XP):
        return self.__Kernel1(X,XP)

    def __CheckAccuracy(self,X,Y):
        v=self.predict(X)
        print 'Percent correct:',100-100*sum(abs(v-Y))/(2*len(Y))


    def __SciKitLearn(self, XTest, YTest, X, Y):
        clf = svm.SVC()
        bt = time.time()
        clf.fit(X, Y)
        print 'SciKit Learn Training Time:', time.time() - bt
        v = clf.predict(XTest)
        print 'SciKit Learn Percent correct:', 100 - 100 * sum(abs(v - YTest)) / (2 * len(YTest))

    def __ImportValidationSet(self):
        data = np.loadtxt("data.csv", delimiter=',')
        X = data[:, :2]
        Y = data[:, 2]
        return X,Y


    # Implement this!
    def fit(self, X, Y):
        print ''
        print ''
        print 'Fitting Budged Kernel Perceptron'
        print '--------------------------------------'
        print '--------------------------------------'
        bt=time.time()
        numSamples=self.numsamples
        beta=self.beta
        N=self.N

        numOfPoints=len(X)

        numOfTrain=int(2./3.*numOfPoints)
        XTest=X[numOfTrain:]
        YTest=Y[numOfTrain:]

        X=X[:numOfTrain]
        Y=Y[:numOfTrain]

        self.X=X
        self.Y=Y


        S=[]; alpha={x:0 for x in xrange(numOfTrain)}

        for num in xrange(numSamples):

            t=random.randint(0,numOfTrain-1)
            Xt=X[t];Yt=Y[t]
            YHatT=0

            for i in S:
                YHatT+=alpha[i]*self.__Kernel(Xt,X[i])

            if Yt*YHatT<=beta:
                S.append(t)
                alpha[t]=Yt

                if len(S)>N:
                    YinS=[]

                    for i in S:
                        YHatTi=0
                        for j in S:
                            YHatTi+=alpha[j]*self.__Kernel(X[i],X[j])

                        YinS.append(Y[i]*(YHatTi-alpha[i]*self.__Kernel(X[i],X[i])))

                    ind=np.argmax(YinS)

                    del S[ind]
                    alpha[ind]=0

        w=[0,0]

        for i in xrange(numOfTrain):
            for j in xrange(len(X[i])):
                w[j] += alpha[i] * X[i, j]

        self.w = w
        print 'Fitting Budged Kernel Perceptron Done!'
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
beta = 20
N = 100
numsamples = 10000

kernel_file_name = 'k.png'
budget_kernel_file_name = 'bk.png'

# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.
k = KernelPerceptron(numsamples)
k.fit(X,Y)
k.visualize(kernel_file_name, width=0, show_charts=False, save_fig=True, include_points=False)

bk = BudgetKernelPerceptron(beta, N, numsamples)
bk.fit(X, Y)
bk.visualize(budget_kernel_file_name, width=0, show_charts=False, save_fig=True, include_points=False)