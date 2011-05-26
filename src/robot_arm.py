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

logger = logging.getLogger('robo' + cur_counter)
hdlr = logging.FileHandler(conf.get_run_log_path('ROBOARM')+'robo' + cur_counter + '.log')
formatter = logging.Formatter('%(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

# Find the paths
command_path = conf.get_command_path()
file_path = conf.get_file_path()+"robo" + cur_counter + ".net"
data_file = conf.get_data_path('ROBOARM')+'combined_robot.data'
test_data_file = conf.get_data_path('ROBOARM')+"combined_robot.data"

MADELON_spec = netspec(file_path, command_path, data_file, test_data_file)

MADELON_spec.num_input_units = 2
MADELON_spec.num_hidden_layers = 1

MADELON_spec.num_output_units = 2
MADELON_spec.hidden_output_weights = 'x0.1:0.1'
MADELON_spec.output_bias = '1'

MADELON_spec.train_range = '1:200'
MADELON_spec.test_range = '401:600'

hw_0 = hidden_weights()
hw_0.index = 0
hw_0.num_units = 16
hw_0.ih = '0.1:0.1'
hw_0.bh = '-'

MADELON_spec.hidden_layer_specs.append(hw_0)

MADELON_spec.model_spec = 'real'
MADELON_spec.noise_level = '0.1:0.1'

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
lambdas = array([1900.0])

fn = lambda x, item, epsilon: util.Gaussian_RBF_lambda(x, item, epsilon, lambdas)
opt = optimize(fn)

opt.bounds = [(105.0, 2005.0)]
opt.num_basis = 100
opt.start_point = [50.0]
opt.maxeval = 100
opt.epsilons =  arange(12.0, 16.0, 0.5)
opt.bf_opt_steps = [20.0]
opt.reinitialize()

# First run
super_transition_steps = 32000

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
MADELON_spec.lf_step = 500
MADELON_spec.window_size = 8
MADELON_spec.epsilon = 0.1

MADELON_spec.repeat_iteration = 1
facility.setup_ceiling()

MADELON_spec.sample_sigmas = True
#MADELON_spec.use_decay = True
#MADELON_spec.negate = True

# Loop
for i in range(100):
    print "Iteration:", i
    logger.info("Iteration: " + str(i))
    facility.opt_iter(opt, logger)
