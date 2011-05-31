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

total = 200
cv_fold = 8

conf = config('path_config.cfg')
cur_counter = str(conf.get_and_set_run_counter())

# --------------- Setup logging ---------------
logger = logging.getLogger('led' + cur_counter)
hdlr = logging.FileHandler(conf.get_run_log_path('LED')+'led' + cur_counter + '.log')
formatter = logging.Formatter('%(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)
# ---------------------------------------------

command_path = conf.get_command_path()

data_file = conf.get_data_path('LED')+'led-noisy.data'
test_data_file = conf.get_data_path('LED')+"led-noisy.data"

os.mkdir(conf.get_file_path("led", cur_counter))

LED_specs = []

for i in range(cv_fold):
    file_path = conf.get_file_path("led", cur_counter) + "/led" + str(i) + ".net"

    LED_spec = netspec(file_path, command_path, data_file, test_data_file)

    LED_spec.num_input_units = 24
    LED_spec.num_hidden_layers = 1

    LED_spec.int_target = 10
    LED_spec.num_output_units = 1

    LED_spec.hidden_output_weights = 'x1:0.2'
    LED_spec.output_bias = '1:0.2'

    LED_spec.input_output_weights = '1:0.2:0.2'

    LED_spec.train_range = '-'+str(i*(total/cv_fold)+1)+':'+str((i+1)*(total/cv_fold))
    LED_spec.test_range = str(i*(total/cv_fold)+1)+':'+str((i+1)*(total/cv_fold))

    hw_0 = hidden_weights()
    hw_0.index = 0
    hw_0.num_units = 8
    hw_0.ih = '1:0.2:0.2'
    hw_0.bh = '1:0.2'

    LED_spec.hidden_layer_specs.append(hw_0)

    LED_spec.model_spec = 'class'

    # Set up neural net
    netspec_cmd = LED_spec.generate_netspec_command()
    #print netspec.to_string(netspec_cmd)
    retcode = subprocess.check_call(netspec_cmd)
    #print 'net-spec reuslt:', retcode


    modelspec_command = LED_spec.generate_modelspec_command()
    #print netspec.to_string(modelspec_command)
    retcode = subprocess.check_call(modelspec_command)
    #print 'model-spec reuslt:', retcode


    dataspec_command = LED_spec.generate_dataspec_command()
    #print netspec.to_string(dataspec_command)
    retcode = subprocess.check_call(dataspec_command)
    #print 'data-spec reuslt:', retcode


    netgen_command = LED_spec.generate_netgen_command()
    #print netspec.to_string(netgen_command)
    retcode = subprocess.check_call(netgen_command)
    #print 'net-gen reuslt:', retcode
    
    LED_specs.append(LED_spec)


# ===========================================================
# Setup optimization
# ===========================================================
pure_bayes = True
if len(sys.argv) > 1:
    pure_bayes = bool(sys.argv[1])
    
if pure_bayes:
    lambdas = array([0.5, 800.0])
else:
    lambdas = array([4900.0])

fn = lambda x, item, epsilon: util.Gaussian_RBF_lambda(x, item, epsilon, lambdas)
opt = optimize(fn)

if pure_bayes:
    opt.bounds = [(0.2, 0.7), (5.0, 805.0)]
    opt.num_basis = 100
    opt.start_point = [0.4, 200.0]
    opt.maxeval = 100
    opt.epsilons =  arange(3.5, 10.0, 0.5)
    opt.bf_opt_steps = [0.05, 50.0]
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
super_transition_steps = 2500

for LED_spec in LED_specs:
    # Starter run setup
    LED_spec.lf_step = 50
    LED_spec.window_size = 5
    LED_spec.epsilon = 0.4

    LED_spec.repeat_iteration = 200
    LED_spec.ceiling = 10
    LED_spec.sample_sigmas = False
    LED_spec.use_decay = False
    LED_spec.negate = False


facility = facilities(super_transition_steps, LED_specs, opt)

# Starter Run
facility.starter_run(logger)


facility.epsilon = 0.1
facility.lf_step = 500
facility.setup_ceiling()

for LED_spec in LED_specs:
    # Final runs setup
    LED_spec.lf_step = facility.lf_step
    LED_spec.window_size = 10
    LED_spec.epsilon = facility.epsilon

    LED_spec.repeat_iteration = 10

    LED_spec.sample_sigmas = True
    #LED_spec.use_decay = True
    #LED_spec.negate = True

# Loop
for i in range(150):
    print "Iteration:", i
    logger.info("Iteration: " + str(i))
    facility.opt_iter(logger)
