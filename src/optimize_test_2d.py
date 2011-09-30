from optimize import *
import util
from numpy import *
from numpy.random import *
import time

def objective_2(x, y):
    #return cos(50*(0.5*exp(-((x-0.3)**2 - ((y-250.0)/4980.0)**2)) + 0.5*exp((-(x-0.5)**2 - ((y-1050.0)/4980.0)**2))))/((x-0.5)**2 + ((y-1050.0)/4980.0)**2+1)
    
    return cos(30*(0.5*exp(-((x-0.3)**2 - ((y-250.0)/4980.0)**2)) + 0.5*exp((-(x-0.5)**2 - ((y-1050.0)/4980.0)**2))))/((x-0.5)**2 + ((y-1050.0)/4980.0)**2+1)
    
def objective(x):
    return objective_2(x[0], x[1])

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

lambdas = array([1.0, 5001.0])

fn = lambda x, item, epsilon: util.Gaussian_RBF_lambda(x, item, epsilon, lambdas)
opt = optimize(fn)

# Simulate the Robo-Arm problem
opt.bounds = [(0.01, 1.01), (20.0, 5021.0)]
opt.num_basis = 300
opt.start_point = [0.4, 200.0]
opt.maxeval = 100
opt.epsilons =  arange(4.0, 6.1, 4.0)
opt.bf_opt_steps = [0.1, 200.0]

# Simulate the Dexter problem
#opt.bounds = [(0.2, 0.7), (50.0, 1500.0)]
#opt.num_basis = 100
#opt.start_point = [0.4, 200.0]
#opt.maxeval = 100
#opt.epsilons =  arange(3.5, 8.0, 0.5)
#opt.bf_opt_steps = [0.1, 100.0]

opt.reinitialize()

x = []
y = []
z = []

pt = opt.start_point
for i in range(40):
    time1 = time.time()
    
    #if (i+1)%10 == 0:
	#opt.resample()
    
    noisy_y = objective(pt) #+ normal(loc=0.0, scale=0.5)
    print i, pt, noisy_y
    opt.update(pt, noisy_y)
    pt = opt.bf_opt(float(i+1)) # Use brute force to optimize
    #pt = opt.direct(float(i+1)) # Use direct to optimize
    
    time2 = time.time()
    print 'Took:', time2-time1, 'secs'
    
    x.append(pt[0])
    y.append(pt[1])
    z.append(noisy_y)


plot = False
if plot:
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
    import matplotlib.pyplot as plt

    contour = True
    
    X = arange(0.0, 1.02, 0.01)
    Y = arange(20, 5000, 50.0)
    X, Y = meshgrid(X, Y)
    Z = objective_2(X, Y)

    if contour == True:
	plt.figure()
	CS = plt.contour(X, Y, Z, 20)
	plt.scatter(x, y)
	plt.clabel(CS, inline=1, fontsize=10)
	plt.title('Simplest default with labels')
    else:
	fig = plt.figure()
	ax = Axes3D(fig)
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
		linewidth=0, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

if not plot:
    import matplotlib.pyplot as plt
    
    X = arange(50, 5000, 50.0)
    y = 0.51
    ucbs, lcbs, mus = twodFunc(X, y)
    
    plt.plot(X, mus, 'r')
    plt.fill_between(X, ucbs, lcbs, color='#0066FF')
    plt.xlabel('L', fontsize=16)
    plt.ylabel('Reward', fontsize=16)
    #plt.axis([50, 5000, 3, 5])
    plt.show()
    
