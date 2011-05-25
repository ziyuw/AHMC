# -*- coding: utf-8 -*-
import subprocess
from control import *
from optimize import *
from numpy import *
import util
from config import *

import logging


conf = config('path_config.cfg')
cur_counter = str(conf.get_and_set_run_counter())

logger = logging.getLogger('madelon' + cur_counter)
hdlr = logging.FileHandler(conf.get_run_log_path('MADELON')+'madelon' + cur_counter + '.log')
formatter = logging.Formatter('%(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

# Find the paths
command_path = conf.get_command_path()
file_path = conf.get_file_path()+"madelon" + cur_counter + ".net"
data_file = conf.get_data_path('MADELON')+'combined.data.sel'

MADELON_spec = netspec(file_path, command_path, data_file)

MADELON_spec.num_input_units = 38
MADELON_spec.num_hidden_layers = 2

MADELON_spec.num_output_units = 1
MADELON_spec.hidden_output_weights = 'x0.1:1:4'
MADELON_spec.output_bias = '10'

MADELON_spec.train_range = '1:10'
MADELON_spec.test_range = '11:20'

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
print netspec.to_string(netspec_cmd)
retcode = subprocess.check_call(netspec_cmd)
print 'net-spec reuslt:', retcode


modelspec_command = MADELON_spec.generate_modelspec_command()
print netspec.to_string(modelspec_command)
retcode = subprocess.check_call(modelspec_command)
print 'model-spec reuslt:', retcode


dataspec_command = MADELON_spec.generate_dataspec_command()
print netspec.to_string(dataspec_command)
retcode = subprocess.check_call(dataspec_command)
print 'data-spec reuslt:', retcode


netgen_command = MADELON_spec.generate_netgen_command()
print netspec.to_string(netgen_command)
retcode = subprocess.check_call(netgen_command)
print 'net-gen reuslt:', retcode


# Setup opt
lambdas = array([0.40, 2000.0])
fn = lambda x, item, epsilon: util.Gaussian_RBF_lambda(x, item, epsilon, lambdas)
opt = optimize(RBF_func = fn)

# First run
super_transition_steps = 10000

# Starter run setup
MADELON_spec.lf_step = 100
MADELON_spec.window_size = 4
MADELON_spec.epsilon = 0.02

MADELON_spec.repeat_iteration = 40
MADELON_spec.ceiling = 10
MADELON_spec.sample_sigmas = False
MADELON_spec.use_decay = False
MADELON_spec.negate = False


facility = facilities(super_transition_steps, MADELON_spec)

# Starter Run
facility.starter_run(logger)


# Final runs setup
MADELON_spec.lf_step = 800
MADELON_spec.window_size = 8
MADELON_spec.epsilon = 0.05

MADELON_spec.repeat_iteration = 1
facility.setup_ceiling()

MADELON_spec.sample_sigmas = True
#MADELON_spec.use_decay = True
MADELON_spec.negate = True

# Loop
for i in range(100):
    print "Iteration:", i
    logger.info("Iteration: " + str(i))
    facility.opt_iter(opt, logger)
