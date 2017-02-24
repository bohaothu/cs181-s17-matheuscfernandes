from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
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

        if isSharedCovariance:
            sigma=(np.cov(X, rowvar=0))
        for k in xrange(numberOfClasses): # looping over class K
            values=X[Y==k,:]
            valuesAll.append(values)
            YY=Y[Y==k]

            pik.append(float(np.sum(YY))/float(np.sum(Y)))
            mu[0,k]=np.mean([values[:,0]])
            mu[1,k] = np.mean([values[:,1]])
            if not isSharedCovariance:
                sigma.append(np.cov(values,rowvar=0))

        if isSharedCovariance:
            BV0 = multivariate_normal(mu[:,0], sigma)
            BV1 = multivariate_normal(mu[:,1], sigma)
            BV2 = multivariate_normal(mu[:,2], sigma)
        else:
            BV0 = multivariate_normal(mu[:, 0], sigma[0])
            BV1 = multivariate_normal(mu[:, 1], sigma[1])
            BV2 = multivariate_normal(mu[:, 2], sigma[2])

        lkh = []
        lkh.append(sum(np.log(BV0.pdf(valuesAll[0])*pik[0])))
        lkh.append(sum(np.log(BV1.pdf(valuesAll[1])*pik[1])))
        lkh.append(sum(np.log(BV2.pdf(valuesAll[2])*pik[2])))
        print lkh


        self.mu=mu # pass back the values of the mean
        self.sigma=sigma # pass back the values of the variance
        self.pik=pik
        return

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
            predict[:,k]=mv.pdf(X_to_predict)# obtain the PDF for a given class for each of the vales of X
        Y=np.argmax(predict,axis=1)

        return np.array(Y)

    # Do not modify this method!
    def visualize(self, output_file, width=3, show_charts=True):
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
