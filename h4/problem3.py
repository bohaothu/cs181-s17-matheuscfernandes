# CS 181, Spring 2017
# Homework 4: Clustering
# Name: Matheus C Fernandes
# Email: fernandes@seas.harvard.edu

import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg


class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K

    def __norm(self,X):
        return sum(sum(X**2.))**(1./2.)

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    def fit(self, X):
        K = self.K

        mu=np.zeros((K,X.shape[1],X.shape[2]))
        lenXi=X.shape[0]

        for k in range(int(K)):
            beginInt = random.randint(0, (X.shape[0]) - 1)
            mu[k,:,:]=X[beginInt,:,:]

        converged=True
        ct=0
        while converged:
            ct+=1
            print'Iteration: ',ct
            r=np.zeros((X.shape[0],K))

            for i in range(lenXi):
                allMu=[]
                for k in range(int(K)):
                    allMu.append(self.__norm(X[i,:,:]-mu[k,:,:]))

                allMu=np.array(allMu)
                r[i,np.argmin(allMu)]=1.

            if ct==1: oldLoss=0
            else: oldLoss=Loss

            Loss=0.

            for k in range(K):
                nk=sum(r[:,k])

                if nk!=0:
                    mu[k,:,:]=0.
                    for i in range(lenXi):
                        mu[k,:,:]+=(1./nk)*r[i,k]*X[i,:,:]
                    for i in range(lenXi):
                        Loss+=r[i,k]*self.__norm(X[i,:,:]-mu[k,:,:])

            deltaLoss=abs(oldLoss-Loss)
            print 'Delta Loss:',deltaLoss
            if deltaLoss<100: converged=False

        self.mu=mu
        self.get_mean_images()

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        return self.mu


    # This should return the arrays for D images from each cluster that are representative of the clusters.
    def get_representative_images(self, D):
        pass

    # img_array should be a 2D (square) numpy array.
    # Note, you are welcome to change this function (including its arguments and return values) to suit your needs.
    # However, we do ask that any images in your writeup be grayscale images, just as in this example.
    def create_image_from_array(self, img_array):
        plt.figure()
        plt.imshow(img_array, cmap='Greys_r')
        plt.show()
        return


# This line loads the images for you. Don't change it!
pics = np.load("images.npy", allow_pickle=False)

# You are welcome to change anything below this line. This is just an example of how your code may look.
# That being said, keep in mind that you should not change the constructor for the KMeans class, 
# though you may add more public methods for things like the visualization if you want.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.

K = 10
KMeansClassifier = KMeans(K)
KMeansClassifier.fit(pics)
KMeansClassifier.create_image_from_array(pics[1])
