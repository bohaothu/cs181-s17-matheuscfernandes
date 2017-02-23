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

        if isSharedCovariance:
            mu=np.zeros((numberOfClasses)) # create a mean vector with means for all of the different classes
            sigma=1 # create a shared covariance value
            pik=np.zeros((numberOfClasses))
            for k in xrange(numberOfClasses): # looping over class K
                bv=multivariate_normal(mu[k],sigma) #create a multivariate distribution for each class with shared variance
                bv=bv.pdf(X[:,k]) # obtain the PDF for a given class for each of the vales of X


        self.mu=mu # pass back the values of the mean
        self.sigma=sigma # pass back the values of the variance

        return

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        Y = []
        for x in X_to_predict:
            val = 0
            if x[1] > 4:
                val += 1
            if x[1] > 6:
                val += 1
            Y.append(val)
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
