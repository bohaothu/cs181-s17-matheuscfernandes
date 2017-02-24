from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
import math
import pandas as pd

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class GaussianGenerativeModel:
    def __init__(self, isSharedCovariance=False):
        self.isSharedCovariance = isSharedCovariance
        self.numberOfClasses=3
        self.iterations=100

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        isSharedCovariance=self.isSharedCovariance
        numberOfClasses = self.numberOfClasses
        iterations=self.iterations

        mu=np.zeros((2,3))
        pik=[];sigma=[];valuesAll=[]

        if isSharedCovariance: sigma=np.zeros((2,2))
        for k in xrange(numberOfClasses): # looping over class K
            values=X[Y==k,:]
            valuesAll.append(values)
            YY=Y[Y==k]

            pik.append(float(len(YY))/float(len(Y)))
            mu[0,k]=np.mean([values[:,0]])
            mu[1,k] = np.mean([values[:,1]])
            if not isSharedCovariance:
                sigma.append(np.cov(values,rowvar=0))
            if isSharedCovariance:
                sigma = sigma+(float(len(values))/float(len(X)))*(np.cov(values, rowvar=0))


        self.mu=mu # pass back the values of the mean
        self.sigma=sigma # pass back the values of the variance
        self.pik=pik
        return

    def likelihood(self):
        mu=self.mu
        sigma=self.sigma
        Y=self.Y
        X=self.X
        C = np.array(pd.get_dummies(Y))
        pik=self.pik
        isSharedCovariance=self.isSharedCovariance

        lkh=0

        for k in xrange(C.shape[1]):
            for i in xrange(C.shape[0]):

                mult=np.asmatrix(X[i,:]-mu[:,k])
                if isSharedCovariance:
                    sigmaInv = np.linalg.inv(sigma)
                    lkh += (np.log(2.*math.pi)+(1./2.)*np.log(np.linalg.det(sigma))+(1./2.)*np.matmul(np.matmul(mult,sigmaInv),mult.T))*C[i,k]-np.log(pik[k])
                else:
                    sigmaInv = np.linalg.inv(sigma[k])
                    lkh += (np.log(2. * math.pi) + (1. / 2.) * np.log(np.linalg.det(sigma[k])) + (1. / 2.) * np.matmul(
                        np.matmul(mult, sigmaInv), mult.T)) * C[i, k] - np.log(pik[k])
        if isSharedCovariance:
            print 'shared covariance likelihood is: '+str(lkh[0,0])+'\n'
        else:
            print 'non-shared covariance likelihood is: '+str(lkh[0,0])+'\n'


    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        isSharedCovariance=self.isSharedCovariance
        mu=self.mu
        pik=self.pik
        sigma=self.sigma
        predict=np.zeros((X_to_predict.shape[0],3))
        for k in xrange(3):
            if not isSharedCovariance:
                mv=multivariate_normal(mu[:,k],sigma[k]) #create a multivariate distribution for each class with shared variance
            if isSharedCovariance:
                mv = multivariate_normal(mu[:, k], sigma)
            predict[:,k]=mv.pdf(X_to_predict)*pik[k]# obtain the PDF for a given class for each of the vales of X
        Y=np.argmax(predict,axis=1)

        return np.array(Y)

    # Do not modify this method!
    def visualize(self, output_file, width=3, show_charts=False):
        X = self.X

        # Create a grid of points
        x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
        y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
        xx,yy = np.meshgrid(np.arange(x_min, x_max, .005), np.arange(y_min,
            y_max, .005))

        # Flatten the grid so the values match spec for self.predict
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        X_topredict = np.vstack((xx_flat,yy_flat)).T

        # Get the class predictions
        Y_hat = self.predict(X_topredict)
        Y_hat = Y_hat.reshape((xx.shape[0], xx.shape[1]))

        cMap = c.ListedColormap(['r','b','g'])

        # Visualize them.
        plt.figure()
        plt.pcolormesh(xx,yy,Y_hat, cmap=cMap)
        plt.scatter(X[:, 0], X[:, 1], c=self.Y, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
