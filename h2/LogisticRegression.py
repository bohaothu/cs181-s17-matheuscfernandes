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
    def __init__(self, eta, lambda_parameter,iterations):
        self.eta = eta
        self.lambda_parameter = lambda_parameter
        self.numberOfClasses=3
        self.iterations=iterations
    
    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None


    # TODO: Implement this method!
    def fit(self, X, C):
        self.X = X
        self.C = C

        lambda_parameter=self.lambda_parameter
        eta = self.eta
        iterations=self.iterations
        numberOfClasses=self.numberOfClasses

        C=np.array(pd.get_dummies(C)) # convert all of the values to one-hot vectors

        Bias=np.ones((C.shape[0])) #create a bias column
        X=np.column_stack((Bias,X)) # stack the bias column

        w=np.ones((X.shape[1],numberOfClasses)) # initialize a weight matrix
        loss=[];itAll=[]
        for it in xrange(iterations):
            wx=np.matmul(X,np.transpose(w))   ### double check this to make sure its correct
            sf= np.exp(wx)  # convert the wx to the exponential form for the softmax function
            for i in xrange(wx.shape[0]): # for all the classes
                sumSf=np.sum(sf[i,:]) # create a sum vector for the soft max
                sf[i,:]=sf[i,:]/sumSf # divide each class by the sum vector for that respective class

            L=np.matmul(np.transpose((sf-C)),X)
            loss.append(-np.sum((C*np.log(sf)))+lambda_parameter*np.sum(w*w))

            w=w-eta*(L+2*lambda_parameter*w)

            itAll.append(it)

        plt.plot(itAll,loss)

        self.weights=w
        return

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        w=self.weights

        Bias = np.ones((X_to_predict.shape[0]))  # create a bias column
        X_to_predict = np.column_stack((Bias, X_to_predict))  # stack the bias column

        Y=np.argmax(np.matmul(X_to_predict,np.transpose(w)),axis=1)

        return Y

    def visualize(self, output_file, width=2, show_charts=True):
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
