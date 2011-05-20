from numpy import *
from numpy.random import *
import scipy.linalg

"""
This file handles Bayesian Linear Regression
with Normal-Inverse-Gamma prior and fixed basis 
given by basis
"""

class BayesLinModel:
    def __init__(self, v_0, w_0, a_0, b_0, epsilon, basis, RBF_func = None):
	"""
	All variables are initialized as matrices except for scalors
	"""
	
	self.basis = basis # This is generated from latin hyper cube
	self.n = 0.0	# The number of points we have seen so far
	self.cov = v_0	# The covariance function
	self.cov_inv = linalg.inv(v_0)	# The inverse of the covariance function
	self.epsilon = epsilon	# The width in RBF
	self.mean = w_0	# The mean
	self.a_0 = float(a_0)
	self.a_n = a_0
	self.b_n = float(b_0)
	self.const_for_b_n = self.mean.H*self.cov_inv*self.mean
	self.const_for_mean = self.cov_inv*w_0
	self.RBF_func = RBF_func
    
    def update(self, x, y_n):
	"""
	Updates the model after seeing a new point
	
	x is a column vector
	y_n (scalar) is the objective value
	"""
	# Change x into features
	x = self.compute_RBF(x)
	
	# Rank 2 update
	self.n = self.n + 1.0
	self.cov = self.cov - (self.cov*x*x.H*self.cov)/(1+x.H*self.cov*x)
	
	# Update cov_inv
	self.cov_inv = self.cov_inv + x*x.H
	
	# Update mean
	self.const_for_mean = self.const_for_mean + y_n*x
	self.mean = self.cov*self.const_for_mean
	
	# update a_n and b_n
	self.a_n = self.a_0 + self.n/2
	self.const_for_b_n = self.const_for_b_n + y_n**2
	self.b_n = self.b_n + 0.5*(self.const_for_b_n - self.mean.H*self.cov_inv*self.mean)
	
	#print self.mean.H*self.cov_inv*self.mean, "product", y_n**2
	#print self.a_n
	#print self.b_n

    def predict(self, x):
	"""
	Returns the mean, variance, and the degree of freedom
	for the multivariate student t distribution
	
	x is a column vector of features
	"""
	
	# NOTE: need to figure out what happens when n = 0
	# The behaviour seems to make sense
	
	# Change x into features
	x = self.compute_RBF(x)
	
	#print x.H*self.cov*x, self.b_n
	
	predict_mean = x.H*self.mean
	predict_var = (self.b_n/self.a_n)*(1.0 + x.H*self.cov*x)
	df = float(2*self.a_n)
	
	return float(predict_mean), float(df*predict_var/(df-2)), df
	
    def compute_RBF(self, x):
	"""
	Compute the Gaussian RBF features given input
	
	x is a column vector
	NOTE: rescale the RBF seems necessary
	"""
	if shape(x)[0] > 1:
	    x = reshape(x, (1, shape(x)[0]))
	size = shape(self.basis)[0]
	
	# Features is a column vector
	features = mat(zeros((size, 1)))
	
	counter = 0
	for item in self.basis:
	    if self.RBF_func == None:
		features[counter] = exp(-(self.epsilon*linalg.norm(x - item))**2)
	    else:
		features[counter] = self.RBF_func(x, item, self.epsilon)
	    counter = counter + 1
	
	return features

class group_linreg:
    def __init__(self, epsilons, v_0, w_0, a_0, b_0, basis, RBF_func = None):
	"""
	epsilons is a list of epsilons with equal probability
	"""
	self.alpha_const = 0.1
	self.delta = 0.05
	self.dim = shape(v_0)[0]
	self.size = size(epsilons)
	self.linreg_list = []
	for epsilon in epsilons:
	    self.linreg_list.append(BayesLinModel(v_0, w_0, a_0, b_0, epsilon, basis, RBF_func))
	    
    def update(self, x, y_n):
	"""
	Updates each linear model after seeing a new point
	
	x is a column vector
	y_n (scalar) is the objective value
	"""
	for lin_model in self.linreg_list:
	    lin_model.update(mat(x), y_n)
	
    def predict(self, x):
	"""
	This returns the expectation and the variance
	"""
	
	means = empty((self.size, 1))
	variances = empty((self.size, 1))
	
	for i in range(self.size):
	    means[i], variances[i], df = self.linreg_list[i].predict(mat(x))
	
	# Total law of Expectation
	predict_mean = mean(means)
	
	#print means
	#print variances
	
	# Total law of Variance
	predict_variance = var(means) + mean(variances)
	
	#print predict_variance
	
	return predict_mean, predict_variance

    def compute_UCB(self, x, t):
	# NOTE: Check this out
	x = array(x); x = mat(reshape(x, (1, shape(x)[0])));
	mean, variance = self.predict(x)
	
	alpha = sqrt(2*log(self.dim*t**2*pi**2/float(6*self.delta)))*self.alpha_const
	return mean + alpha*variance
