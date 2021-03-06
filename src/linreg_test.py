import numpy
from linreg import *
from util import *
from matplotlib.pyplot import *

dim = 100
bounds = [(2.0,7.0)]
basis = lhsample(dim, bounds)
v_0 = mat(eye(dim))
w_0 = mat(empty((dim, 1)))
a_0 = 0.1
b_0 = 0.1
epsilon = 3.0

# NOTE: How to choose a_0 and b_0?
# NOTE: This can be chosen in smart ways probably

epsilons =  arange(2.0, 5.0, 1)

linearmodel = group_linreg(epsilons, v_0, w_0, a_0, b_0, basis)
#linearmodel = BayesLinModel(v_0, w_0, a_0, b_0, epsilon, basis)

seq = arange(2.0, 7.0, 0.1)
#y_value = -(0.5*seq*seq*seq - 0.1*seq*seq)
y_value = 2*numpy.cos(seq*4)

for i in range(size(seq)):
    linearmodel.update(mat([seq[i]]), y_value[i])

print "Training finished."

# Testing
test_range = arange(2.0, 7.0, 0.1)
predictions = []
upper = []
lower = []
for pt in test_range:
    mu, sigma = linearmodel.predict(mat(pt))
    predictions.append(mu)
    upper.append(mu+sigma)
    lower.append(mu-sigma)


#print predictions
line1 = plot(seq, y_value)
line2 = plot(test_range, predictions, '-ro')
line3 = plot(test_range, upper)
plot(test_range, lower)
figlegend( (line1, line2, line3),
           ('label1', 'label2', 'label3'),
           'upper right' )
show()