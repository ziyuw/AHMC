from numpy import *
import numpy.random as nr

import ego_ctypes as ego_c
from ego.gaussianprocess import GaussianProcess
from ego.gaussianprocess.kernel import GaussianKernel_ard
from ego.acquisition import EI, UCB, maximizeEI, maximizeUCB

import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import matplotlib.pyplot as plt

class GPBO:
    def __init__(self, bound, kernel_hyperparms):
	self.counter = 0
	self.bound = bound
	self.kernel_hyperparms = kernel_hyperparms
	self.kernel = GaussianKernel_ard(self.kernel_hyperparms)
	self.gp = GaussianProcess(self.kernel, noise = 0.01)
	self.all_params = []; self.responses = []
	self.annealing_schedule = lambda t: exp(-0.01*t)
	pass
    
    def bf_opt(self, nothing):
	prob = self.annealing_schedule(self.counter - 1)
	if nr.rand() < prob:
	    _, params = maximizeUCB(self.gp, self.bound, delta=0.1, scale=prob)
	else:
	    params = self.all_params[-1]
	return params
    
    def update(self, params, response):
	self.counter = self.counter + 1
	self.gp.addData(params, response)
	self.all_params.append(params)
	self.responses.append(response)
	pass
    
    def name_it_later(self, X1, X2):
	m = zeros_like(X1)
	v = zeros_like(X1)
	for i in xrange(X1.shape[0]):
	    for j in xrange(X1.shape[1]):
		z = array([X1[i, j], X2[i, j]])
		res = self.gp.posterior(z)
		m[i, j] = res[0]
		v[i, j] = res[1]
	return m
    
    def plot_response_surface(self, contour=True):
	X = arange(self.bound[0][0], self.bound[0][1], \
	    (self.bound[0][1]-self.bound[0][0])/50.0)
	Y = arange(self.bound[1][0], self.bound[1][1], \
	    (self.bound[1][1]-self.bound[1][0])/50.0)
	X, Y = meshgrid(X, Y)
	Z = self.name_it_later(X, Y)
	x = [item[0] for item in self.all_params]
	y = [item[1] for item in self.all_params]
	z = self.responses
	
	plt.clf()	
	if contour == True:
	    plt.figure(1)
	    CS = plt.contour(X, Y, Z, 20)
	    plt.scatter(x, y)
	    plt.clabel(CS, inline=1, fontsize=10)
	    plt.xlabel('Step size adjustment', fontsize = 16)
	    plt.ylabel('No. of leapfrog steps', fontsize = 16)
	    plt.show()
	else:
	    fig = plt.figure(1)
	    ax = Axes3D(fig)
	    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, \
		cmap=cm.jet, linewidth=0, antialiased=False)
	    ax.scatter(x, y, z)
	    ax.set_xlabel('Step size adjustment', fontsize = 16)
	    ax.set_ylabel('No. of leapfrog steps', fontsize = 16)
	    ax.set_zlabel('Reward', fontsize = 16)
	    fig.colorbar(surf, shrink=0.5, aspect=5)
	    plt.show()
	    
	
    def slice_plot(self):
	axis = 0
	X = arange(self.bound[axis][0], self.bound[axis][1], \
	    (self.bound[axis][1]-self.bound[axis][0])/100.0)
	y = 1500
	prob = self.annealing_schedule(self.counter - 1)
	ucb = UCB(self.gp, len(self.bound), delta=0.1, scale=prob)
	
	
	mean = []
	var = []
	for x in X:
	    if axis == 0:
		z = array([x, y])
	    elif axis == 1:
		z = array([y, x])
		
	    res = self.gp.posterior(z)
	    mean.append(res[0])
	    var.append(res[1])
	
	plt.figure(1)
	plt.subplot(211)
	
	plt.plot(X, mean, 'r')
	plt.fill_between(X, mean+sqrt(var), mean-sqrt(var), color='#0066FF')
	
	plt.axis([self.bound[axis][0], self.bound[axis][1], min(mean-sqrt(var))-0.05, max(mean+sqrt(var))+0.05])
	plt.xlabel('Step Size Adjustment', fontsize=16)
	plt.ylabel('Reward', fontsize=16)
	
	acqs = mean+self.annealing_schedule(self.counter - 1)*ucb.sBeta*sqrt(var)
	
	plt.subplot(212)
	plt.fill_between(X, acqs, 0, color='#99CC99')
	plt.axis([self.bound[axis][0], self.bound[axis][1], min(acqs), max(acqs)+0.5])
	plt.xlabel('Step Size Adjustment', fontsize=16)
	plt.ylabel('Acquisition Function Value', fontsize=16)
	plt.show()
	
	
	
	
