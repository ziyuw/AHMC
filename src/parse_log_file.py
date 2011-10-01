from optimize import *
import util
from numpy import *
from numpy.random import *
import time

def func(X, Y):
    mus = zeros(X.shape)
    for i in range(X.shape[0]):
	for j in range(X.shape[1]):
	    pt_in_func = [X[i,j], Y[i,j]]
	    pt_in_func = mat(reshape(pt_in_func, (1, shape(pt_in_func)[0])));
	    mus[i,j], sigma = opt.predict(pt_in_func)
    return mus

def twodFunc(X, y, flip = False):
    ucb = zeros(X.shape)
    lcb = zeros(X.shape)
    mus = zeros(X.shape)
    acq = zeros(X.shape)
    for i in xrange(X.shape[0]):
	if not flip:
	    pt = [X[i], y]
	else:    
	    pt = [y, X[i]]
	mus[i], sigma = opt.predict(pt)
	ucb[i] = mus[i] + sqrt(sigma)
	lcb[i] = mus[i] - sqrt(sigma)
	acq[i] = opt.acquisition(pt)
    return ucb, lcb, mus, acq

#file_path = "temp.txt"
#file_path = "dexter105.log"
#file_path = "madelon16.log"
file_path = "robo120.log"
#file_path = "madelon3.log"

f = open(file_path, 'r')

pt = [0.1, 500]
reward = None

lambdas = array([1.0, 5001.0])

fn = lambda x, item, epsilon: util.Gaussian_RBF_lambda(x, item, epsilon, lambdas)
opt = optimize(fn)

#opt.bounds = [(0.2, 0.6), (50.0, 5000.0)]
#opt.num_basis = 200
#opt.start_point = [0.4, 300.0]
#opt.maxeval = 100
#opt.epsilons =  arange(2.5, 3.0, 0.5)
#opt.bf_opt_steps = [0.05, 100.0]

#opt.bounds = [(0.2, 0.7), (50.0, 1500.0)]
#opt.num_basis = 100
#opt.start_point = [0.4, 200.0]
#opt.maxeval = 100
#opt.epsilons =  arange(0.5, 6.0, 0.5)
#opt.bf_opt_steps = [0.05, 50.0]

# Madelon
#opt.bounds = [(0.3, 0.9), (50.0, 2000.0)]
#opt.num_basis = 200
#opt.start_point = [0.4, 200.0]
#opt.maxeval = 100
#opt.epsilons =  arange(3.5, 10.0, 0.5)
#opt.bf_opt_steps = [0.05, 50.0]

# Robo old
#opt.bounds = [(0.01, 1.01), (20.0, 5000.0)]
#opt.num_basis = 300
#opt.start_point = [0.4, 200.0]
#opt.maxeval = 100
#opt.epsilons =  arange(3.5, 10.0, 0.5)
#opt.bf_opt_steps = [0.02, 20.0]

# Robo Sept. 29th. 2011
opt.bounds = [(0.01, 1.01), (20.0, 5021.0)]
opt.num_basis = 300
opt.start_point = [0.4, 200.0]
opt.maxeval = 100
opt.epsilons =  arange(10.0, 22.1, 4.0)
opt.bf_opt_steps = [0.02, 100.0]

# Dexter
#opt.bounds = [(0.05, 0.6), (50.0, 2000.0)]
#opt.num_basis = 200
#opt.start_point = [0.4, 200.0]
#opt.maxeval = 100
#opt.epsilons =  arange(3.5, 10.0, 0.5)
#opt.bf_opt_steps = [0.05, 50.0]

opt.reinitialize()

x = []
y = []
z = []

for line in f:
    if 'Reward' in line:
	reward = float(line.split(':')[1].strip())
    elif 'New params:' in line:
	line_spt = line.split(':')[1].split()
	pt = []
	pt.append(float(line_spt[0]))
	pt.append(float(line_spt[1]))
    
    if pt != None and reward != None:
	print pt, reward
	x.append(pt[0])
	y.append(pt[1])
	z.append(reward)
	opt.update(pt, reward)
	pt = None
	reward = None

plot = False
contour = True
if plot:
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    
    
    X = arange(0.01, 1.01, 0.02)
    Y = arange(20, 5021, 50.0)
    X, Y = meshgrid(X, Y)
    Z = func(X, Y)

    if contour == True:
	plt.figure()
	CS = plt.contour(X, Y, Z)
	plt.clabel(CS, inline=1, fontsize=16)
	plt.title('Simplest default with labels')
    else:
	fig = plt.figure()
	ax = Axes3D(fig)
	
	ax.scatter(x, y, z)
	
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
		linewidth=0, antialiased=False)
	ax.set_xlabel('Step size adjustment', fontsize = 16)
	ax.set_ylabel('No. of leapfrog steps', fontsize = 16)
	ax.set_zlabel('Reward', fontsize = 16)
	fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    
twodplot = True
if twodplot:
    import matplotlib.pyplot as plt
    
    #doflip = True
    #X = arange(20.0, 5021.0, 20.0)
    #y = 0.4
    
    doflip = False
    X = arange(0.01, 1.01, 0.01)
    y = 3520
    
    ucbs, lcbs, mus, acqs = twodFunc(X, y, flip = doflip)
    
    plt.subplot(211)
    plt.plot(X, mus, 'r')
    plt.fill_between(X, ucbs, lcbs, color='#0066FF')
    
    plt.xlabel('Step Size Adjustment', fontsize=16)
    plt.ylabel('Reward', fontsize=16)
    plt.axis([0.01, 1.01, -1, 1])
    
    plt.subplot(212)
    plt.fill_between(X, acqs, 0, color='#99CC99')
    plt.xlabel('Step Size Adjustment', fontsize=16)
    plt.ylabel('Acquisition Function Value', fontsize=16)
    plt.axis([0.01, 1.01, 0, 0.2])
    plt.show()
    