import subprocess
from collections import deque
from numpy import *
from numpy.random import *
from config import *

def generate_netpred_command(ran, file_path, option, command_path, test_data_path):
    """
    generate command for net-pred
    
    net-pred itn rlog.net 101
    """
    cmd = []; cmd.append(command_path+'net/net-pred')
    cmd.append(option)
    cmd.append(file_path)
    cmd.append(str(ran)); cmd.append('/')
    cmd.append(test_data_path); cmd.append('.')
    
    return cmd

def class_err(result):
    result = result[0].split('\n')
    #splited = result.split('\n')
    for line in result:
	if 'Fraction of guesses that were wrong' in line:
	    splitted = line.split()
	    return float(splitted[len(splitted)-1].split('+-')[0])

def num_samples(result):
    result = result[0].split('\n')
    #splited = result.split('\n')
    for line in result:
	if 'Number of iterations used' in line:
	    splitted = line.split()
	    return float(splitted[1])

def write_in_file(data_file_name, start, finish, num_folds):

    conf = config('path_config.cfg')

    command_path = conf.get_command_path()

    test_data_path = conf.get_data_path('DEXTER') + data_file_name #+"test.data.sel"#+"combined_valid.data.sel"#'combined_valid.data.sel'

    net_folder = conf.get_file_path("dexter", cur_counter)
    option = 'am'

    ran = str(start)+":"+str(finish)
    
    numsamples = 0

    ls = []
    for i in range(num_folds):
	net_path = net_folder + '/dexter' + str(i) + '.net'
	cmd = generate_netpred_command(ran, net_path, option, command_path, test_data_path)
	process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE)
	result = process.communicate()
	
	ls.append(class_err(result))
	numsamples = num_samples(result)
    
    return mean(ls), var(ls), ls, num_samples

cur_counter = '124'
start = 1200;
num_folds = 10

f = open('myownfile', 'w')    

# Determine total number of samples
m, v, ls, num_samples = write_in_file("combined_valid.data.sel", start, "", 1)

jump_size = 300
for i in range(2):
     
    finish = max(start+(i+1)*jump_size, num_samples)
    m, v, ls,  = write_in_file("combined_valid.data.sel", start, finish, num_folds)
    print m, v, ls
    f.write( str((i+1)*jump_size)+ " " + str(m) + " " + str(v) + str(ls) +"\n")
