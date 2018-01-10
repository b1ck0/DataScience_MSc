# -*- coding: utf-8 -*-
"""
Spyder Editor
@author : Vasil Yordanov
"""

import numpy as np

def gradient_descent(alpha, x, y, numIterations):
    """
    Implementation of the gradient descent optimization algorithm used for regression
        - cost function is least squares
        - hypothesis function is linear in form theta[1]*x+theta[0]

    inputs:
        x - vector of independent variables [x1,x2,x3,...]
        y - vector of dependent variables [y1,y2,y3,...]
        alpha - learning rate (no solid guidance how to choose this parameter, I usually start with 0.01)
        numIterations - number of maximum iterations to run in case convergence is not reached
        
    output:
        theta[0] - intercept coefficient for linear regression
        theta[1] - slope coefficient for linear regression
    """
    m = x.size                                      # obtaining the number of samples
    x = np.c_[ np.ones(m), x]                       # converting x into a matrix (first column is only 1s and second our transponsed x), so we can utilize numpy
    theta = [0,0]                                   # setting theta[0] and theta[1] to 0
    
    for iter in range(0, numIterations):
        hypothesis = np.dot(x, theta)               # hypothesis function
        error = hypothesis - y                      # error between hypothesis and actual values
        J = np.sum(error ** 2) / (2 * m)            # cost function
        print("iter %s | J: %.3f" % (iter, J))  
        gradient = np.dot(x.transpose(), error) / m # derivatives for both theta[0] and theta[1]
        theta = theta - alpha * gradient            # update theta vector based on the fuction derivatives
    return theta

from sklearn.datasets.samples_generator import make_regression 
import pylab

if __name__ == '__main__':

    x, y = make_regression(n_samples=100, n_features=1, n_informative=1, random_state=0, noise=35) 
    
    theta = gradient_descent(0.01, x, y, 10000)

    for i in range(x.size):
        y_predict = theta[0] + theta[1]*x 
    
    pylab.plot(x,y,'o')
    pylab.plot(x,y_predict,'x-')
    pylab.show()
    print("Done!")

def func(x):
    return x**2 + 3*x - 4
    
def gradient_descent_optimization(alpha = 0.01, function, function_gradient, x0, bounds, numIterations = 1000):
    """
    Implementation of the gradient descent optimization algorithm. 

    inputs:
       alpha - learning rate (no solid guidance how to choose this parameter, I usually start with 0.01)
       x0 - observations for our independent variables
       function - definition of the function which we want to optimize
       function_gradient - definition of the derivatives of the function which we are optimizing
       bounds - limits for our independent variables
       numIterations - limit for our iterations when our algorithm will be killed if it is not converged by then
      
    output:
       x1 - vector of the optimal values of our independent variables
    """
    x1 = np.zeros(x0)                               # setting x1 to zero vector with length equal to the length of x0
    
    return x1