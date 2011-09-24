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

def twodFunc(X, y):
    ucb = zeros(X.shape)
    lcb = zeros(X.shape)
    mus = zeros(X.shape)
    
    for i in xrange(X.shape[0]):
	pt = [y, X[i]]
	mus[i], sigma = opt.predict(pt)
	ucb[i] = mus[i] + sqrt(sigma)
	lcb[i] = mus[i] - sqrt(sigma)
    return ucb, lcb, mus

#file_path = "temp.txt"
#file_path = "dexter105.log"
#file_path = "madelon16.log"
file_path = "robo111.log"
#file_path = "madelon3.log"

f = open(file_path, 'r')

pt = [0.1, 500]
reward = None

lambdas = array([0.55, 1950.0])

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
<<<<<<< HEAD
#opt.num_basis = 200
#opt.start_point = [0.4, 200.0]
#opt.maxeval = 100
#opt.epsilons =  arange(3.5, 10.0, 0.5)
#opt.bf_opt_steps = [0.05, 50.0]
=======
#opt.num_basis = 200
#opt.start_point = [0.4, 200.0]
#opt.maxeval = 100
#opt.epsilons =  arange(3.5, 10.0, 0.5)
#opt.bf_opt_steps = [0.05, 50.0]

#opt.bounds = [(0.2, 0.6), (50.0, 5000.0)]
#opt.num_basis = 200
#opt.start_point = [0.4, 200.0]
#opt.maxeval = 100
#opt.epsilons =  arange(0.5, 4.0, 0.5)
#opt.bf_opt_steps = [0.05, 100.0]

>>>>>>> 34dcf318b245779036c975acb85e3e808d24c832

# Robo
opt.bounds = [(0.2, 0.6), (50.0, 5000.0)]
opt.num_basis = 200
opt.start_point = [0.4, 200.0]
opt.maxeval = 100
opt.epsilons =  arange(3.5, 10.0, 0.5)
opt.bf_opt_steps = [0.05, 100.0]

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
	# For robo arm
	#reward = reward/1000.0 + 0.99
	
	
	print pt, reward
	x.append(pt[0])
	y.append(pt[1])
	z.append(reward)
	opt.update(pt, reward)
	pt = None
	reward = None

plot = False
contour = False
if plot:
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    
    contour = False
    
<<<<<<< HEAD
    X = arange(0.2, 0.6, 0.05)
    Y = arange(50, 5000, 50.0)
=======
    X = arange(0.05, 0.6, 0.02)
    Y = arange(50, 2000, 20.0)
>>>>>>> 34dcf318b245779036c975acb85e3e808d24c832
    X, Y = meshgrid(X, Y)
    Z = func(X, Y)

    if contour == True:
	plt.figure()
	CS = plt.contour(X, Y, Z)
	plt.clabel(CS, inline=1, fontsize=10)
	plt.title('Surface learned by Bayesian ')
    else:
	fig = plt.figure()
	ax = Axes3D(fig)
	
	scatter = ax.scatter(x, y, z, label='Samples drawn')
	
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
		linewidth=0, antialiased=False, label='Surface learned by contexual bandits')
	fig.colorbar(surf, shrink=0.5, aspect=5)
	
	
	ax.set_xlabel('Step size adjustment', fontsize=16)
	ax.set_ylabel('No. of leapfrog steps', fontsize=16)
	ax.set_zlabel('Reward', fontsize=16)
	
	#plt.rcParams['label.fontsize'] = 10
	
	# For robot arm
	#ax.set_zlim3d(0.98, 1.00)

    plt.show()
    
twodplot = True
if twodplot:
    import matplotlib.pyplot as plt
    
    X = arange(50, 5000, 50.0)
    y = 0.35
    ucbs, lcbs, mus = twodFunc(X, y)
    
    plt.plot(X, mus, 'r')
    plt.fill_between(X, ucbs, lcbs, color='#0066FF')
    plt.xlabel('L', fontsize=16)
    plt.ylabel('Reward', fontsize=16)
    
    plt.show()
    
