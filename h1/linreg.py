#####################
# CS 181, Spring 2016
# Homework 1, Problem 3
#
##################

import csv
import seaborn
import numpy as np
import matplotlib.pyplot as plt

# Creating the different basis functions according to the PSET
def p3a(times): return np.ones(times.shape),times,times**2,times**3,times**4,times**5,times**6
def p3b(times): return np.ones(times.shape),times,times**2,times**3,times**4
def p3c(times): return np.ones(times.shape),np.sin(times/1.),np.sin(times/2.),np.sin(times/3.),np.sin(times/4.),np.sin(times/5.),np.sin(times/6.)
def p3d(times): return np.ones(times.shape),np.sin(times/1.),np.sin(times/2.),np.sin(times/3.),np.sin(times/4.),np.sin(times/5.),np.sin(times/6.),np.sin(times/7.),np.sin(times/8.),np.sin(times/9.),np.sin(times/10.)
def p3e(times): return np.ones(times.shape),np.sin(times/1.),np.sin(times/2.),np.sin(times/3.),np.sin(times/4.),np.sin(times/5.),np.sin(times/6.),np.sin(times/7.),np.sin(times/8.),np.sin(times/9.),np.sin(times/10.),np.sin(times/11.),np.sin(times/12.),np.sin(times/13.),np.sin(times/14.),np.sin(times/15.),np.sin(times/16.),np.sin(times/17.),np.sin(times/18.),np.sin(times/19.),np.sin(times/20.),np.sin(times/21.),np.sin(times/22.)


# Change for a particular part of the problem
def basis(times): return p3e(times)

csv_filename = 'congress-ages.csv'
times  = []
ages = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        times.append(float(row[0]))
        ages.append(float(row[1]))

# Turn the data into numpy arrays.
times  = np.array(times)
ages = np.array(ages)

# Plot the data.
plt.plot(times, ages, 'o')
plt.xlabel("Congress age (nth Congress)")
plt.ylabel("Average age")
# plt.show()

# Create the simplest basis, with just the time and an offset.

X = np.vstack((basis(times))).T

# Nothing fancy for outputs.
Y = ages

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_times!!!!!
grid_times = np.linspace(75, 120, 200)
grid_X = np.vstack(basis(grid_times))
grid_Yhat  = np.dot(grid_X.T, w)

# Plot the data and the regression line.
plt.plot(times, ages, 'o', grid_times, grid_Yhat, '-')
plt.xlabel("Congress age (nth Congress)")
plt.ylabel("Average age")
plt.show()
# plt.savefig('images/p3e.pdf', bbox_inches='tight')


