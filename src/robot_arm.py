# -*- coding: utf-8 -*-
import subprocess
from control import *
from optimize import *
from numpy import *
import util
from config import *

import logging
import sys
import os

cv_fold = 8
total = 200

conf = config('path_config.cfg')
cur_counter = str(conf.get_and_set_run_counter())

# --------------- Setup logging ---------------
logger = logging.getLogger('robo' + cur_counter)
hdlr = logging.FileHandler(conf.get_run_log_path('ROBOARM')+'robo' + cur_counter + '.log')
formatter = logging.Formatter('%(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)
# ---------------------------------------------

data_file = conf.get_data_path('ROBOARM')+'combined_robot.data.train'
test_data_file = conf.get_data_path('ROBOARM')+"combined_robot.data.train"

os.mkdir(conf.get_file_path("robo", cur_counter))

command_path = conf.get_command_path()

ROBO_specs = []

for i in range(cv_fold):
    # Find the paths
    file_path = conf.get_file_path("robo", cur_counter) + "/" + "robo" + str(i) + ".net"

    ROBO_spec = netspec(file_path, command_path, data_file, test_data_file)

    ROBO_spec.num_input_units = 2
    ROBO_spec.num_hidden_layers = 1

    ROBO_spec.num_output_units = 2
    ROBO_spec.hidden_output_weights = 'x0.1:0.1'
    ROBO_spec.output_bias = '1'

    ROBO_spec.train_range = '-'+str(i*(total/cv_fold)+1)+':'+str((i+1)*(total/cv_fold))
    ROBO_spec.test_range = str(i*(total/cv_fold)+1)+':'+str((i+1)*(total/cv_fold))

    hw_0 = hidden_weights()
    hw_0.index = 0
    hw_0.num_units = 16
    hw_0.ih = '0.1:0.1'
    hw_0.bh = '0.1:0.1'

    ROBO_spec.hidden_layer_specs.append(hw_0)

    ROBO_spec.model_spec = 'real'
    ROBO_spec.noise_level = '0.1:0.1'

    # Set up neural net
    netspec_cmd = ROBO_spec.generate_netspec_command()
    #print netspec.to_string(netspec_cmd)
    retcode = subprocess.check_call(netspec_cmd)
    #print 'net-spec reuslt:', retcode


    modelspec_command = ROBO_spec.generate_modelspec_command()
    #print netspec.to_string(modelspec_command)
    retcode = subprocess.check_call(modelspec_command)
    #print 'model-spec reuslt:', retcode


    dataspec_command = ROBO_spec.generate_dataspec_command()
    #print netspec.to_string(dataspec_command)
    retcode = subprocess.check_call(dataspec_command)
    #print 'data-spec reuslt:', retcode


    netgen_command = ROBO_spec.generate_netgen_command()
    #print netspec.to_string(netgen_command)
    retcode = subprocess.check_call(netgen_command)
    #print 'net-gen reuslt:', retcode

    ROBO_specs.append(ROBO_spec)

# ===========================================================
# Setup optimization
# ===========================================================
pure_bayes = True
if len(sys.argv) > 1:
    pure_bayes = bool(sys.argv[1])
    
if pure_bayes:
    lambdas = array([0.4, 4950.0])
else:
    lambdas = array([4900.0])

fn = lambda x, item, epsilon: util.Gaussian_RBF_lambda(x, item, epsilon, lambdas)
opt = optimize(fn)

if pure_bayes:
    opt.bounds = [(0.2, 0.6), (50.0, 5000.0)]
    opt.num_basis = 200
    opt.start_point = [0.4, 200.0]
    opt.maxeval = 100
    opt.epsilons =  arange(0.5, 4.0, 0.5)
    opt.bf_opt_steps = [0.02, 100.0]
else:
    opt.bounds = [(105.0, 5005.0)]
    opt.num_basis = 200
    opt.start_point = [50.0]
    opt.maxeval = 100
    opt.epsilons =  arange(12.0, 16.0, 0.5)
    opt.bf_opt_steps = [20.0]
opt.reinitialize()

# ===========================================================
# Setup optimization end
# ===========================================================

# Set up the number of super transition steps
super_transition_steps = 32000

for ROBO_spec in ROBO_specs:
    # Starter run setup
    ROBO_spec.lf_step = 100
    ROBO_spec.window_size = 1
    ROBO_spec.epsilon = 0.02

    ROBO_spec.repeat_iteration = 40
    ROBO_spec.ceiling = 10
    ROBO_spec.sample_sigmas = False
    ROBO_spec.use_decay = False
    ROBO_spec.negate = False


facility = facilities(super_transition_steps, ROBO_specs, opt)

# Starter Run
facility.starter_run(logger)

facility.epsilon = 0.1
facility.lf_step = 500
facility.setup_ceiling()

for ROBO_spec in ROBO_specs:
    # Final runs setup
    ROBO_spec.lf_step = facility.lf_step
    ROBO_spec.window_size = 10
    ROBO_spec.epsilon = facility.epsilon

    ROBO_spec.repeat_iteration = 1

    ROBO_spec.sample_sigmas = True
    #ROBO_spec.use_decay = True
    #ROBO_spec.negate = True

# Loop
for i in range(100):
    print "Iteration:", i
    logger.info("Iteration: " + str(i))
    facility.opt_iter(logger)
