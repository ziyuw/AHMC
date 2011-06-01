import subprocess
from collections import deque
from numpy import *
from numpy.random import *
from config import *

def generate_netpred_command(discard, file_path, option, command_path, test_data_path):
    """
    generate command for net-pred
    
    net-pred itn rlog.net 101
    """
    cmd = []; cmd.append(command_path+'net/net-pred')
    cmd.append(option)
    cmd.append(file_path)
    cmd.append(str(discard)+':'); cmd.append('/')
    cmd.append(test_data_path); cmd.append('.')
    
    return cmd
    
def parse_result(result):
    result = result[0].split()
    l = []
    for line in result:
	l.append(int(line))
    return l

conf = config('path_config.cfg')

command_path = conf.get_command_path()

test_data_path = conf.get_data_path('DEXTER')+"combined_valid.data.sel"#'combined_valid.data.sel'

cur_counter = '105'
net_folder = conf.get_file_path("dexter", cur_counter)
option = 'bm'

num_folds = 10

ls = []
for i in range(num_folds):
    net_path = net_folder + '/dexter' + str(i) + '.net'
    cmd = generate_netpred_command(11, net_path, option, command_path, test_data_path)
    
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

print len(result)

f = open('./dexter_valid.resu', 'w')
for item in result:
    f.write(item+"\n")
f.close()