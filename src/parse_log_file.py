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

#file_path = "robo61.log"
file_path = "temp.txt"

f = open(file_path, 'r')

pt = [0.1, 500]
reward = None

lambdas = array([0.5, 1950.0])

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

opt.bounds = [(0.3, 0.8), (50.0, 2000.0)]
opt.num_basis = 200
opt.start_point = [0.4, 200.0]
opt.maxeval = 100
opt.epsilons =  arange(7.5, 8.0, 0.5)
opt.bf_opt_steps = [0.05, 50.0]

#opt.bounds = [(0.2, 0.6), (50.0, 5000.0)]
#opt.num_basis = 200
#opt.start_point = [0.4, 200.0]
#opt.maxeval = 100
#opt.epsilons =  arange(6.5, 7.0, 0.5)
#opt.bf_opt_steps = [0.05, 100.0]

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

print x
print y
print z

plot = True
contour = False
if plot:
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
    import matplotlib.pyplot as plt

    contour = False
    
    X = arange(0.3, 0.8, 0.05)
    Y = arange(50, 2000, 50.0)
    X, Y = meshgrid(X, Y)
    Z = func(X, Y)

    if contour == True:
	plt.figure()
	CS = plt.contour(X, Y, Z)
	plt.clabel(CS, inline=1, fontsize=10)
	plt.title('Simplest default with labels')
    else:
	fig = plt.figure()
	ax = Axes3D(fig)
	
	ax.scatter(x, y, z)
	
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
		linewidth=0, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()