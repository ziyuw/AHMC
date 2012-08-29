from optimize import *
import util
from numpy import *
from numpy.random import *
from gpbo import *
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

file_path = "../run_log/dexter178.log"
file_path = "../run_log/robo177.log"
f = open(file_path, 'r')

pt = [0.1, 500]
reward = None

# Dexter
alpha = 0.2
bound = bounds = [(0.01, 0.61), (20.0, 2001.0)]

# ========================
# Robo-arm
# ========================
alpha = 0.2
bound = [(0.01, 1.01), (20.0, 5021.0)]
# ========================


kernel_hyperparms = array([(bound[0][1] - bound[0][0])*alpha, \
			(bound[1][1] - bound[1][0])*alpha])
opt = GPBO(bound, kernel_hyperparms)
opt.start_point = [0.4, 200.0]



x = []
y = []
z = []

first = True
super_size = 10000
num_iter = 0
max_iter = 500
num_total_samples = super_size/opt.start_point[1]

for line in f:
    if 'Iteration' in line:
	cur_iter = int(line.split(':')[1].strip())
	if cur_iter >= max_iter:
	    break
	
    if 'step_size:' in line:
	num_samples = int(line.split(':')[1].strip())
	num_total_samples = num_total_samples + num_samples
    
    if 'Reward' in line and not first:
	reward = float(line.split(':')[1].strip())
    elif 'Reward' in line and first:
	first = False
	reward = float(line.split(':')[1].strip())
	pt = opt.start_point

    if 'New params:' in line:
	line_spt = line.split(':')[1].split()
	pt = []
	pt.append(float(line_spt[0]))
	pt.append(float(line_spt[1]))
    
    if pt != None and reward != None:
	print cur_iter, pt, reward
	x.append(pt[0])
	y.append(pt[1])
	z.append(reward)
	opt.update(pt, reward)
	pt = None
	reward = None

opt.plot_response_surface(contour=True)
opt.slice_plot()

