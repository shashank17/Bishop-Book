#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:14:57 2017

@author: ShashankAvusali
"""
import numpy as np
import matplotlib.pyplot as plt
import math


# In[1]
def get_sample_data(nsamples):
    step = 1/nsamples
    x = np.arange(0,1,step)
#    x = np.random.uniform(0,1,nsamples)
    y = np.sin(2*np.pi*x) +  np.random.normal(0,0.5,nsamples)
    return x,y

def get_original(minx, maxx):
    x = np.arange(minx,maxx,0.001)
    y = np.sin(2*np.pi*x)
    return x,y

def estimate_params(trainx, trainy, order, l = 0):
    #AW = B
    print(l)
    A = np.array([np.array([np.sum(trainx**(j+i)) for j in np.arange(0,order+1)]) for i in np.arange(0,order+1)])
    B = np.array([np.sum(np.multiply(trainy, trainx**i)) for i in np.arange(0,order+1)]).T
    
    W = np.dot(np.linalg.inv(A+l*np.eye(order+1)),B)
    return W

def plot_poly(w,minx, maxx):
    x = np.arange(minx,maxx,0.001)
    y = predict(x,w)
    plt.plot(x,y,'b-')

def predict(x,w):
    order = w.size - 1
    y = np.array([np.dot(w,np.array([i**j for j in np.arange(0,order+1)]).T) for i in x])
    return y

def calculate_residual_error(x,y,w):
    esty = predict(x,w)
    error = 0.5*np.sum((esty - y)**2)
    return error

def get_rms(x,y,w):
    N = x.size
    res_error = calculate_residual_error(x,y,w)
    rms = np.sqrt(2*res_error/N)
    return rms
    

# In[2]
np.random.seed(1000)

# In[3] Un-regularized linear regression
nsamples_train = 10
nsamples_test = 100
M = 9

trainx, trainy = get_sample_data(nsamples_train)
w = estimate_params(trainx, trainy, M)
print(w)
minx = np.min(trainx)
maxx = np.max(trainx)
underlyingx, underlyingy = get_original(minx, maxx)

plot_poly(w,minx, maxx)
plt.plot(underlyingx, underlyingy,'g-')
plt.plot(trainx,trainy,'ro')
plt.show()

#testx, testy = get_sample_data(nsamples_test)
#max_order = 10
#train_rms = []
#test_rms = []
#for order in range(0,max_order):
#    w = estimate_params(trainx, trainy, order)
#    train_rms.append(get_rms(trainx, trainy, w))
#    test_rms.append(get_rms(testx, testy, w))
#    
#order = np.arange(0,max_order, 1)
#plt.plot(order, train_rms, 'bo-')
#plt.plot(order, test_rms, 'ro-')
    
# In[4] Regularized linear regression
w = estimate_params(trainx, trainy, M, math.exp(-10))
print(w)

plot_poly(w,minx, maxx)
plt.plot(underlyingx, underlyingy,'g-')
plt.plot(trainx,trainy,'ro')
plt.show()