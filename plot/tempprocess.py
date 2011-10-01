import subprocess
from collections import deque
from numpy import *
from numpy.random import *
from matplotlib.pyplot import *

x = load("./tempfile.npy")

print x.shape

num_fold = 8
num_samples = 4916
start = 1500
jump = 50
steps = (num_samples-start)/jump + 1

y = []
z = []

for item in x:
    y.append(item[0])
    z.append(sqrt(item[1]))
    print sqrt(item[1])

a = []
jump_size = 50
for i in range(steps):
    finish = start + (i+1)*jump
    a.append(finish)

x = mat(x)


errorbar(a, y, z, fmt='o')

xlabel('Number of Samples' , fontsize=16)
ylabel('Mean Squred Error', fontsize=16)


show()
