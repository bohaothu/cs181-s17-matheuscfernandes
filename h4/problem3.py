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
    def __init__(self, K, D):
        self.K = K
        self.D = D

    def __norm(self, X):
        return np.linalg.norm(X)  # np.sum(np.sum(X**2.))**(1./2.)

    def varyingK(self, X, numberOfRestarts, ks):
        for k in ks:
            self.K = k
            self.random_Restarts(X, numberOfRestarts)

    def random_Restarts(self, X, numberOfRestarts):
        Loss = [];
        ct = []
        for r in xrange(numberOfRestarts):
            self.randomR = r
            Loss.append(self.fit(X))
            ct.append(r)
        plt.figure()
        plt.plot(ct, Loss, lw=3)
        plt.title(r'Loss vs. RandomRestart K={}'.format(int(self.K)))
        plt.xlabel(r'Random Restart Number')
        plt.ylabel(r'Final Loss')
        # plt.savefig("plots/RR-K{}.pdf".format(int(self.K)), dpi=100)

        self.plot_representative_images(self.D, X)
        self.plot_Loss()

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    def fit(self, X):
        K = self.K

        mu = np.zeros((K, X.shape[1], X.shape[2]))
        lenXi = X.shape[0]

        # initialize the variable mu within random points of X
        for k in xrange(int(K)):
            beginInt = random.randint(0, X.shape[0] - 1)
            mu[k, :, :] = X[beginInt, :, :]

        # usefull definitions for convergence
        converged = True
        ct = 0
        Loss = []
        ctAll = []

        print 'Fitting Began K = {}, Random Iteration = {}'.format(self.K, self.randomR + 1)

        while converged:
            ct += 1
            r = np.zeros((X.shape[0], K))  # resetting the R matrix to have all 0's

            # obtaining the R-matrix based on the differences between mu and the X's
            for i in xrange(lenXi):
                allMu = np.array([self.__norm(X[i, :, :] - mu[k, :, :]) for k in range(int(K))])
                r[i, np.argmin(allMu)] = 1.

            LossSum = 0
            # updating the mean matrices
            for k in xrange(K):
                nk = sum(r[:, k])
                if nk != 0:
                    mu[k] = np.sum([(1. / nk) * r[i, k] * X[i, :, :] for i in range(lenXi)], axis=0)
                # calculating the loss based on the equation in the slides
                LossSum += np.sum([r[i, k] * self.__norm(X[i, :, :] - mu[k, :, :]) for i in range(lenXi)])

            # updating master list of all loss and counts
            Loss.append(LossSum)
            ctAll.append(ct)

            # printing status and checking convergence for breaking while loop
            if ct != 1 and abs(Loss[-1] - Loss[-2]) < 1: converged = False; print '\nConverged!\n'

        # exporting data/information to self
        self.Loss = np.array(Loss)
        self.ctAll = np.array(ctAll)
        self.mu = mu

        self.plot_representative_images(self.D, X)
        self.plot_Loss()

        return min(Loss)


    def plot_Loss(self):
        plt.figure()
        plt.plot(self.ctAll, self.Loss, lw=3)
        plt.title(r'Loss vs. Iteration K={}'.format(self.K))
        plt.xlabel(r'Iteration')
        plt.ylabel(r'Loss')
        plt.savefig("plots/L-K{}-R{}.pdf".format(int(self.K),int(self.randomR)), dpi=100)

    def plot_representative_images(self, D, X):
        import matplotlib.gridspec as gridspec

        muImages = self.mu
        RepImages = self.get_representative_images(D, X)

        fig = plt.figure()
        gs = gridspec.GridSpec(D + 1, muImages.shape[0])
        ax = [plt.subplot(gs[i]) for i in xrange((D + 1) * (muImages.shape[0]))]

        gs.update(hspace=0)

        ctr = 0
        for muCount in xrange(muImages.shape[0]):
            ax[ctr].imshow(muImages[muCount].reshape(28, 28), cmap='Greys_r')
            ax[ctr].axis('off')
            ctr += 1
        for DCount in xrange(D):
            for muCount in xrange(muImages.shape[0]):
                ax[ctr].imshow(RepImages[muCount, DCount, :, :].reshape(28, 28), cmap='Greys_r')
                ax[ctr].axis('off')
                ctr += 1
        plt.suptitle(r'K={} and D={}'.format(int(muImages.shape[0]), int(D)), size=20)
        plt.savefig("plots/KD-K{}-R{}.pdf".format(int(self.K),int(self.randomR)), dpi=100)

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        return self.mu

    # This should return the arrays for D images from each cluster that are representative of the clusters.
    def get_representative_images(self, D, X):
        mu = self.mu
        ImageDiff = np.zeros((mu.shape[0], X.shape[0]))
        ImageClosest = np.zeros((mu.shape[0], D, X.shape[1], X.shape[2]))

        for k in xrange(mu.shape[0]):
            ImageDiff[k] = [self.__norm(X[i, :, :] - mu[k, :, :]) for i in range(X.shape[0])]
            SortIndex = np.array([i[0] for i in sorted(enumerate(ImageDiff[k]), key=lambda x: x[1])])
            ImageClosest[k] = X[SortIndex[:D], :, :]

        return ImageClosest

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
D = 10
numberOfRestarts=3
KMeansClassifier = KMeans(K,D)
# KMeansClassifier.varyingK(pics,numberOfRestarts,[5,10,15])
KMeansClassifier.random_Restarts(pics, numberOfRestarts)