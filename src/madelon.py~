import subprocess
from control import *
from config_setup import *
from optimize import *
from numpy import *
import util

import logging


logger = logging.getLogger('madelon')
hdlr = logging.FileHandler('./madelon2.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

# Find the paths
command_path = get_command_path()
file_path = get_file_path()+"templog.net"
data_file = get_data_path('MADELON')+'combined.data.sel'

MADELON_spec = netspec(file_path, command_path, data_file)

MADELON_spec.num_input_units = 38
MADELON_spec.num_hidden_layers = 2

MADELON_spec.num_output_units = 1
MADELON_spec.hidden_output_weights = 'x0.1:1:4'
MADELON_spec.output_bias = '10'

MADELON_spec.train_range = '1:1900'
MADELON_spec.test_range = '1901:2000'

hw_0 = hidden_weights()
hw_0.index = 0
hw_0.num_units = 20
hw_0.ih = '0.1:1:2'
hw_0.bh = '0.2:1'

hw_1 = hidden_weights()
hw_1.index = 1
hw_1.num_units = 8
hw_1.hh = 'x0.3:1'
hw_1.bh = '0.2:1'

MADELON_spec.hidden_layer_specs.append(hw_0)
MADELON_spec.hidden_layer_specs.append(hw_1)

# Set up neural net
netspec_cmd = MADELON_spec.generate_netspec_command()
retcode = subprocess.check_call(netspec_cmd)
print 'net-spec reuslt:', retcode

netspec_cmd_str = netspec.to_string(netspec_cmd)
print netspec_cmd_str


modelspec_command = MADELON_spec.generate_modelspec_command()
#print modelspec_command
retcode = subprocess.check_call(modelspec_command)
print 'model-spec reuslt:', retcode

dataspec_command = MADELON_spec.generate_dataspec_command()
#print dataspec_command
retcode = subprocess.check_call(dataspec_command)
print 'data-spec reuslt:', retcode

netgen_command = MADELON_spec.generate_netgen_command()
#print netgen_command
retcode = subprocess.check_call(netgen_command)
print 'net-gen reuslt:', retcode


# Setup opt
lambdas = array([0.29, 2000])
fn = lambda x, item, epsilon: util.Gaussian_RBF_lambda(x, item, epsilon, lambdas)
opt = optimize(RBF_func = fn)

# First run
super_transition_steps = 20000

MADELON_spec.lf_step = 100
MADELON_spec.window_size = 10
MADELON_spec.epsilon = 0.10


facility = facilities(super_transition_steps, MADELON_spec)

# Loop
for i in range(100):
    print "Iteration:", i
    facility.opt_iter(opt, logger)

