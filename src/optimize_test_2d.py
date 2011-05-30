from optimize import *
import util
from numpy import *
from numpy.random import *

def objective_2(x, y):
    return cos(30*(0.5*exp(-((x-0.3)**2 - ((y-250.0)/1450.0)**2)) + 0.5*exp((-(x-0.5)**2 - ((y-1050.0)/1450.0)**2))))/((x-0.5)**2 + ((y-1050.0)/1450.0)**2+1)
    
def objective(x):
    return objective_2(x[0], x[1])


plot = False
if plot:
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
    import matplotlib.pyplot as plt

    contour = False
    
    X = arange(-0.2, 0.7, 0.02)
    Y = arange(50, 1500, 20.0)
    X, Y = meshgrid(X, Y)
    Z = objective_2(X, Y)

    if contour == True:
	plt.figure()
	CS = plt.contour(X, Y, Z)
	plt.clabel(CS, inline=1, fontsize=10)
	plt.title('Simplest default with labels')
    else:
	fig = plt.figure()
	ax = Axes3D(fig)
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
		linewidth=0, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

lambdas = array([0.5, 1450.0])

fn = lambda x, item, epsilon: util.Gaussian_RBF_lambda(x, item, epsilon, lambdas)
opt = optimize(fn)

#opt.bounds = [(0.2, 0.6), (50.0, 5000.0)]
#opt.num_basis = 200
#opt.start_point = [0.4, 300.0]
#opt.maxeval = 100
#opt.epsilons =  arange(2.5, 3.0, 0.5)
#opt.bf_opt_steps = [0.05, 100.0]

opt.bounds = [(0.2, 0.7), (50.0, 1500.0)]
opt.num_basis = 100
opt.start_point = [0.4, 200.0]
opt.maxeval = 100
opt.epsilons =  arange(0.5, 6.0, 0.5)
opt.bf_opt_steps = [0.05, 50.0]

opt.reinitialize()

x = opt.start_point
for i in range(100):
    noisy_y = objective(x) + normal(loc=0.0, scale=0.1)
    print i, x, noisy_y
    opt.update(x, noisy_y)
    x = opt.bf_opt(float(i+1)) # Use brute force to optimize
    # x = opt.direct(float(i+1)) # Use direct to optimize
