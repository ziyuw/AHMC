import subprocess
from collections import deque
from numpy import *
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
    result = result.split()
    for line in result:
	print line

conf = config('path_config.cfg')

command_path = conf.get_command_path()

test_data_path = conf.get_data_path('DEXTER')+"combined_valid.data.sel"#'combined_valid.data.sel'

cur_counter = '105'
net_folder = conf.get_file_path("dexter", cur_counter)
option = 'bm'

num_folds = 10
for i in range(num_folds):
    net_path = net_folder + 'dexter' + str(i) + '.net'
    cmd = generate_netpred_command(11, net_path, option, command_path, test_data_path)
    
    process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE)
    result = process.communicate()
    
    parse_result(result)

