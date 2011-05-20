from numpy import *
from numpy.random import *

def lhsample(N, bounds):
    """ 
    Return N samples from a latin hypercube with the given bounds.
    
    The return value is an (N,D) array where D corresponds to the number of 
    bounds given.
    
    Example: print lhsample(10, [(10.0,20.0),(-10.0,-3.0)])
    """
    
    D = len(bounds)
    sample = vstack(arange(a,b,(b-a)/N) for (a,b) in bounds).T + rand(N,D) / N 
    for d in xrange(D): 
        shuffle(sample[:,d])
    return sample
    
    
def multitpdf(x, df, mu, sigma, d):
    """
    Calculate the density of multivariate student T distribution
    """
    const = divide(scipy.special.gamma(float(df + d)/2), scipy.special.gamma(float(df)/2))
    return const*power(det(sigma)*power(df*pi, d), -0.5)*power(1 + 1.0/df*(x-mu).H*inv(sigma)*(x-mu), -float(df+d)/2)

def Gaussian_RBF_lambda(x, item, epsilon, lambdas):
    return exp(-(epsilon*linalg.norm((x - item)/lambdas))**2)
    