from optimize import *
import util
from numpy import *
from numpy.random import *
from matplotlib.pyplot import *

def objective(seq):
    return float(0.5*log(seq)/6.0*cos(seq[0]/50.0)+6.0)

lambdas = array([1000])

fn = lambda x, item, epsilon: util.Gaussian_RBF_lambda(x, item, epsilon, lambdas)
opt = optimize(fn)

#opt.bounds = [(2.0, 7.0)]
#opt.num_basis = 50
#opt.start_point = [3.0]
#opt.maxeval = 100
#opt.epsilons =  arange(15.0, 16.0, 0.5)
#opt.bf_opt_steps = [0.1]
#opt.reinitialize()

opt.bounds = [(5.0, 1005.0)]
opt.num_basis = 100
opt.start_point = [50.0]
opt.maxeval = 100
opt.epsilons =  arange(12.0, 18.0, 0.5)
opt.bf_opt_steps = [10.0]
opt.reinitialize()

seq = arange(5, 1001.0, 1.0)
#y_value = -(0.5*seq*seq*seq - 0.1*seq*seq)
y_value = 0.5*log(seq)/6.0*cos(seq/50.0)+6.0

xs = []
ys = []
x = opt.start_point
for i in range(40):
    # x = opt.direct(float(i+1)) # Use direct to optimize
    
    noisy_y = objective(x)# + normal(loc=0.0, scale=0.1)
    print i, x, noisy_y
    xs.append(x)
    ys.append(noisy_y)
    opt.update(x, noisy_y)
    x = opt.bf_opt(float(i+1)) # Use brute force to optimize
    
print "Training finished."

# Testing
test_range = arange(5.0, 1005.0, 10.0)
predictions = []
upper = []
lower = []
for pt in test_range:
    mu, sigma = opt.predict(mat(pt))
    predictions.append(mu)
    upper.append(mu+sigma)
    lower.append(mu-sigma)


#print predictions
line1 = plot(seq, y_value)
line2 = plot(test_range, predictions, '-r*')
line3 = plot(test_range, upper, '--')
line4 = plot(test_range, lower, '-.k')
line5 = plot(xs, ys, 'bo')

figlegend( (line1, line2, line3, line4, line5),
           ('True Function', 'Predictions (After Opt.)', 'Upper Confidence Bound', 'Lower Confidence Bound', 'Sampled points (During Opt.)'),
           'lower right' )
show()