import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
import pandas as pd
from scipy.misc import logsumexp

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class LogisticRegression:
    def __init__(self, eta, lambda_parameter):
        self.eta = eta
        self.lambda_parameter = lambda_parameter
    
    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None


    # TODO: Implement this method!
    def fit(self, X, C):
        self.X = X
        self.C = C
        lambda_parameter=self.lambda_parameter


        C=pd.get_dummies(C)
        Bias=np.ones((C.shape[0]))

        X=np.column_stack((Bias,X))

        w=np.ones((3,3))
        wx=np.dot(X,w)   ### double check this to make sure its correct
        sf=np.zeros((wx.shape[0],wx.shape[1])) #softmax matrix
        for k in xrange(wx.shape[1]): # for all the classes
            for i in xrange(wx.shape[0]): # for all the given data
                sf[i,k]=np.exp(wx[i,k])
            sumSf=np.sum(sf[:,k]) # create a sum vector for the soft max
            sf[:,k]=sf[:,k]/sumSf # divide each class by the sum vector for that respective class

        L=np.transpose(np.dot(np.transpose(X),sf-C))
        L=L+lambda_parameter*w

        self.weights=w
        return

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        w=self.weights

        Y = []
        for x in X_to_predict:
            val = 0
            if x[1] > 4:
                val += 1
            if x[1] > 6:
                val += 1
            Y.append(val)
        return np.array(Y)

    def visualize(self, output_file, width=2, show_charts=False):
        X = self.X

        # Create a grid of points
        x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
        y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
        xx,yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min,
            y_max, .05))

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
        plt.scatter(X[:, 0], X[:, 1], c=self.C, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
