from numpy import *
from numpy.random import *
from statistics import *
from special_funcs import *

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

def student_t_cdf(x, mu, sigma, df):
    return StudentTCDF(df, float(x-mu)/sigma)
    
def student_t_pdf_mod(x, mu, sigma, df):
    
    const = exp(gammln(float(df + 1)/2)-gammln(float(df)/2))/sigma
    const = const*(float(df)*pi)**(-0.5)
    
    return const*(float(1+((x-mu)/sigma)**2/float(df)))**(-(float(df-1)/2))
    
    #const = gammln(float(df + 1)/2)-gammln(float(df)/2) - log(sigma)
    #const = const - 0.5*log(float(df)*pi)
    

def Gaussian_RBF_lambda(x, item, epsilon, lambdas):
    return exp(-(epsilon*linalg.norm((x - item)/lambdas))**2)

def compute_lambdas(bounds):
    lambdas = []
    for bound in bounds:
	single_lambda = bounds[1]-bounds[0]
	lambdas.append(single_lambda)

    return array(lambdas)