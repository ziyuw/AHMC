from numpy import *
import numpy

data_in = open('/home/ziyuw/projects/AHMC/Data/BostonHousing/housing.data', 'r')
data_out_file = open('/home/ziyuw/projects/AHMC/Data/BostonHousing/housing.data.norm', 'w')


data = numpy.loadtxt(data_in)

print data.shape

mean_dev = [(mean(data[0:, i]), sqrt(var(data[0:, i]))) for i in range(data.shape[1])]

data_out = zeros((data.shape))

for i in range(data.shape[1]):
    for j in range(data.shape[0]):
	data_out[j,i] = (data[j, i] - mean_dev[i][0])/mean_dev[i][1]

print data_out

print [(mean(data_out[0:, i]), sqrt(var(data_out[0:, i]))) for i in range(data_out.shape[1])]

numpy.savetxt(data_out_file, data_out)