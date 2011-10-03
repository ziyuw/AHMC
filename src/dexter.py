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

cv_fold = 10
total = 300

conf = config('path_config.cfg')
cur_counter = str(conf.get_and_set_run_counter())

logger = logging.getLogger('dexter' + cur_counter)
hdlr = logging.FileHandler(conf.get_run_log_path('DEXTER')+'dexter' + cur_counter + '.log')
formatter = logging.Formatter('%(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

command_path = conf.get_command_path()

data_file = conf.get_data_path('DEXTER')+'combined_train.data.sel'
test_data_file = conf.get_data_path('DEXTER')+"combined_train.data.sel"#'combined_valid.data.sel'

os.mkdir(conf.get_file_path("dexter", cur_counter))

DEXTER_specs = []
for i in range(cv_fold):
    # Find the paths
    file_path = conf.get_file_path("dexter", cur_counter) + "/" + "dexter" + str(i) + ".net"

    DEXTER_spec = netspec(file_path, command_path, data_file, test_data_file)

    DEXTER_spec.num_input_units = 295
    DEXTER_spec.num_hidden_layers = 2

    DEXTER_spec.num_output_units = 1
    DEXTER_spec.hidden_output_weights = 'x0.1:1:4'
    DEXTER_spec.output_bias = '10'
    
    DEXTER_spec.init_value = 0.25

    DEXTER_spec.train_range = '-'+str(i*(total/cv_fold)+1)+':'+str((i+1)*(total/cv_fold))
    DEXTER_spec.test_range = str(i*(total/cv_fold)+1)+':'+str((i+1)*(total/cv_fold))

    DEXTER_spec.int_target = 2

    hw_0 = hidden_weights()
    hw_0.index = 0
    hw_0.num_units = 20
    hw_0.ih = '0.05:1:2'
    hw_0.bh = '0.2:1'

    hw_1 = hidden_weights()
    hw_1.index = 1
    hw_1.num_units = 8
    hw_1.hh = 'x0.3:1'
    hw_1.bh = '0.2:1'

    DEXTER_spec.hidden_layer_specs.append(hw_0)
    DEXTER_spec.hidden_layer_specs.append(hw_1)

    # Set up neural net
    netspec_cmd = DEXTER_spec.generate_netspec_command()
    print netspec.to_string(netspec_cmd)
    retcode = subprocess.check_call(netspec_cmd)
    #print 'net-spec reuslt:', retcode


    modelspec_command = DEXTER_spec.generate_modelspec_command()
    print netspec.to_string(modelspec_command)
    retcode = subprocess.check_call(modelspec_command)
    #print 'model-spec reuslt:', retcode


    dataspec_command = DEXTER_spec.generate_dataspec_command()
    print netspec.to_string(dataspec_command)
    retcode = subprocess.check_call(dataspec_command)
    #print 'data-spec reuslt:', retcode


    netgen_command = DEXTER_spec.generate_netgen_command()
    print netspec.to_string(netgen_command)
    retcode = subprocess.check_call(netgen_command)
    #print 'net-gen reuslt:', retcode

    DEXTER_specs.append(DEXTER_spec)

# ===========================================================
# Setup optimization
# ===========================================================
pure_bayes = True
if len(sys.argv) > 1:
    pure_bayes = bool(sys.argv[1])
    
if pure_bayes:
    lambdas = array([0.6, 1981.0])
else:
    lambdas = array([1000.0])

fn = lambda x, item, epsilon: util.Gaussian_RBF_lambda(x, item, epsilon, lambdas)
opt = optimize(fn)

if pure_bayes:
    #Old June 2011
    #opt.bounds = [(0.05, 0.6), (50.0, 2000.0)]
    #opt.num_basis = 200
    #opt.start_point = [0.4, 200.0]
    #opt.maxeval = 100
    #opt.epsilons =  arange(3.5, 10.0, 0.5)
    #opt.bf_opt_steps = [0.05, 50.0]
    
    # Sept 30th, 2011
    opt.bounds = [(0.01, 0.61), (20.0, 2001.0)]
    opt.num_basis = 300
    opt.start_point = [0.4, 200.0]
    opt.maxeval = 100
    opt.epsilons =  arange(10.0, 22.1, 4.0)
    opt.bf_opt_steps = [0.03, 60.0]
    
    opt.a_0 = 3.0
    opt.b_0 = 1.0
    #opt.lamda = 3.0
else:
    opt.bounds = [(10.0, 1010.0)]
    opt.num_basis = 500
    opt.start_point = [50.0]
    opt.maxeval = 100
    opt.epsilons =  arange(13.0, 18.0, 0.5)
    opt.bf_opt_steps = [10.0]
opt.reinitialize()

# ===========================================================
# Setup optimization end
# ===========================================================

# First run
super_transition_steps = 10000
#super_transition_steps = 32000

for DEXTER_spec in DEXTER_specs:
    # Starter run setup
    DEXTER_spec.lf_step = 100
    DEXTER_spec.window_size = 4
    DEXTER_spec.epsilon = 0.02

    DEXTER_spec.repeat_iteration = 40
    DEXTER_spec.ceiling = 10
    DEXTER_spec.sample_sigmas = False
    DEXTER_spec.use_decay = False
    DEXTER_spec.negate = False


facility = facilities(super_transition_steps, DEXTER_specs, opt, pure_bayes)

# Starter Run
facility.starter_run(logger)

facility.epsilon = 0.1
facility.lf_step = 500
facility.setup_ceiling()

for DEXTER_spec in DEXTER_specs:
    # Final runs setup
    DEXTER_spec.lf_step = 500
    DEXTER_spec.window_size = 8
    DEXTER_spec.epsilon = 0.1

    DEXTER_spec.repeat_iteration = 1

    DEXTER_spec.sample_sigmas = True
    DEXTER_spec.use_decay = False
    DEXTER_spec.negate = True

# Loop
for i in range(80):
    print "Iteration:", i
    logger.info("Iteration: " + str(i))
    facility.opt_iter(logger)
