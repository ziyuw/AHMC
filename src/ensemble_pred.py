import subprocess
from collections import deque
from numpy import *
from numpy.random import *
from config import *
import os

def generate_netpred_command(start, finish, file_path, option, command_path, test_data_path):
    """
    generate command for net-pred
    
    net-pred itn rlog.net 101
    """
    cmd = []; cmd.append(command_path+'net/net-pred')
    cmd.append(option)
    cmd.append(file_path)
    cmd.append(str(start)+':'); cmd.append('/')
    cmd.append(test_data_path)
    
    return cmd
    
def parse_result(result):
    result = result[0].split()
    l = []
    for line in result:
	l.append(int(line))
    return l


def write_in_file(data_file_name, result_file_name):

    conf = config('path_config.cfg')

    command_path = conf.get_command_path()

    test_data_path = conf.get_data_path('DEXTER') + data_file_name 

    cur_counter = '178'
    net_folder = conf.get_file_path("dexter", cur_counter)
    option = 'bm'

    num_folds = 10

    ls = []
    for i in range(num_folds):
	net_path = net_folder + '/dexter' + str(i) + '.net'
	cmd = generate_netpred_command(6000, 7603, net_path, option, \
	    command_path, test_data_path)
	
	process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE)
	result = process.communicate()
	
	ls.append(parse_result(result))
    
    result = []
    dim = len(ls[0])
    for i in range(dim):
	s = sum([ls[j][i] for j in range(len(ls))])
	if s > len(ls) - s:
	    result.append('1')
	elif s < len(ls) - s:
	    result.append('-1')
	else:
	    if random() > 0.5:
		result.append('1')
	    else:
		result.append('-1')

    f = open(result_file_name, 'w')
    for item in result:
	f.write(item+"\n")
    f.close()
    
    save_individual_results(ls, result_file_name)

def save_individual_results(ls, result_file_name):
    for idx, result in enumerate(ls):
	directory = './dexter_result/dexter_{0}/'.format(idx)
	if not os.path.exists(directory):
	    os.makedirs(directory)
	f = open(directory+result_file_name, 'w')
	for item in result:
	    f.write(str(item*2-1)+"\n")
	f.close()

if __name__ == "__main__":
    
    write_in_file("test.data.sel", './dexter_test.resu')
    write_in_file("combined_valid.data.sel", './dexter_valid.resu')
    write_in_file("combined_train.data.sel", './dexter_train.resu')
