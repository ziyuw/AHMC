# -*- coding: utf-8 -*-
import subprocess
from collections import deque
from numpy import *

class hidden_weights:
    def __init__(self):
	self.index = 0
	self.num_units = 0
	self.hh = '-'
	self.ih = '-'
	self.bh = '-'
	self.th = '-'
    
    def generate(self):
	cmd = []
	if not self.index == 0:
	    cmd.append(self.hh)
	cmd.append(self.ih)
	cmd.append(self.bh)
	cmd.append(self.th)
	
	return cmd

# ===============================================================================================

class netspec:
    def __init__(self, file_path, command_path, data_file, test_data_file):
	self.file_path = file_path # The location that stores the log file
	self.command_path = command_path
	self.num_input_units = 0; self.input_offset = '-'
	self.num_hidden_layers = 0; self.hidden_layer_specs = []
	self.num_output_units = 0; self.hidden_output_weights = '-'
	self.input_output_weights = '-'; self.output_bias = '-'
	
	self.data_file = data_file; self.train_range = ''
	self.test_data_file = test_data_file
	self.test_range = ''; self.model_spec = 'binary'
	
	self.noise_level = ''; self.int_target = 0
	
	self.init_value = 0.5; self.repeat_iteration = 5
	
	self.lf_step = 1000; self.window_size = 8
	self.epsilon = 0.25; self.ceiling = 100
	
	self.sample_sigmas = True; self.decay = '0.8'
	self.negate = True; self.use_decay = True
	
    def make_string(self, cmd):
	for i in range(size(cmd)):
	    cmd[i] = str(cmd[i])
	return cmd
	
    def generate_netspec_command(self):
	cmd = []; cmd.append(self.command_path+'net/net-spec')
	cmd.append(self.file_path); cmd.append(self.num_input_units)
	for hl in self.hidden_layer_specs:
	    cmd.append(hl.num_units)
	if self.int_target != 0 and self.model_spec == 'class':
	    cmd.append(self.int_target)
	else:
	    cmd.append(self.num_output_units)
	cmd.append('/')
	cmd.append(self.input_offset)
	for hl in self.hidden_layer_specs:
	    cmd.extend(hl.generate())
	cmd.append(self.hidden_output_weights)
	cmd.append(self.input_output_weights)
	cmd.append(self.output_bias)
	
	return self.make_string(cmd)
    
    def generate_modelspec_command(self):
	"""
	Generate the command for model-spec
	model-spec blog.net binary
	"""
	cmd = []
	
	cmd.append(self.command_path+'util/model-spec'); cmd.append(self.file_path)
	cmd.append(self.model_spec)
	if self.model_spec == 'real':
	    cmd.append(self.noise_level)
	
	return self.make_string(cmd)
	
    def generate_dataspec_command(self):
	"""
	Generate the command for data-spec
	
	data-spec rlog.net 1 1 / rdata@1:100 . rdata@101:200 .
	"""
	cmd = []; cmd.append(self.command_path+'util/data-spec')
	cmd.append(self.file_path)
	cmd.append(self.num_input_units)
	cmd.append(self.num_output_units)
	if self.int_target != 0:
	    cmd.append(self.int_target)
	cmd.append('/')
	cmd.append(self.data_file+'@'+self.train_range)
	cmd.append('.')
	cmd.append(self.test_data_file+'@'+self.test_range)
	cmd.append('.')
	
	return self.make_string(cmd)
	
    def generate_netgen_command(self):
	"""
	Generate the command for net-gen
	
	net-gen rlog.net fix 0.5
	"""
	cmd = []; cmd.append(self.command_path+'net/net-gen')
	cmd.append(self.file_path); cmd.append('fix')
	cmd.append(self.init_value)
	
	return self.make_string(cmd)

    def generate_mcspec_command(self):
	"""
	Generate the command for mc-spec
	
	mc-spec blog.net repeat 10 sample-noise heatbath hybrid 100:10 0.2
	mc-spec $log repeat 5 sample-sigmas heatbath 0.8 hybrid 800:8 0.05 negate
	"""
	cmd = []; cmd.append(self.command_path+'mc/mc-spec')
	cmd.append(self.file_path); cmd.append('repeat')
	cmd.append(self.repeat_iteration)
	
	if self.sample_sigmas:
	    cmd.append('sample-sigmas')
	
	cmd.append('heatbath')
	
	if self.use_decay:
	    cmd.append(self.decay)
	
	cmd.append('hybrid')
	cmd.append(str(self.lf_step) + ":" + str(self.window_size))
	cmd.append(self.epsilon); 
	
	if self.negate:
	    cmd.append('negate')
	
	return self.make_string(cmd)
	
    def generate_net_display_command(self, index):
	"""
	Generate the command for net-display
	
	net-display madelon0.net 2:
	"""
	cmd = []; cmd.append(self.command_path+'net/net-display')
	cmd.append('-p'); cmd.append(self.file_path);
	cmd.append(str(index))
	
	return self.make_string(cmd)
	
    def generate_netmc_command(self):
	"""
	Generate the command for net-mc
	
	net-mc rlog.net 1
	"""
	cmd = []; cmd.append(self.command_path+'net/net-mc')
	cmd.append(self.file_path); cmd.append(self.ceiling)
	
	return self.make_string(cmd)
    
    def generate_netpred_command(self, discard):
	"""
	generate command for net-pred
	
	net-pred itn rlog.net 101
	"""
	cmd = []; cmd.append(self.command_path+'net/net-pred')
	if self.model_spec == 'binary':
	    cmd.append('tmp')
	elif self.model_spec == 'real':
	    cmd.append('tn')
	elif self.model_spec == 'class':
	    cmd.append('am')
	cmd.append(self.file_path)
	cmd.append(str(discard)+':');
	
	return self.make_string(cmd)
    
    def generate_accpt_command(self):
	"""
	Generate command that can be used for acceptence rate
	
	net-plt t r madelon0.net
	"""
	cmd = []; cmd.append(self.command_path+'net/net-plt')
	cmd.append('t'); cmd.append('r'); cmd.append(self.file_path)
	
	return self.make_string(cmd)
    
    @staticmethod
    def to_string(cmd):
	cmd_string = ''
	for i in range(size(cmd)):
	    cmd_string = cmd_string + cmd[i] + ' '
	return cmd_string

# ===============================================================================================

class facilities:
    
    def __init__(self, super_transition_steps, spec, opt):
	self.super_transition_steps = super_transition_steps
	self.spec = spec
	self.iter_ct = 0
	self.anneal_const = 0.05
	self.last_probs = deque([])
	self.num_probs = 4
	self.prob_thresh = 0.01
	self.opt = opt
	self.reinitialized = False
	self.pure_bayes = True
	self.abandon_model = True
    
# -----------------------------------------------------------------------------------------------
    def update_last_probs(self, prob):
	self.last_probs.append(prob)
	if len(self.last_probs) == self.num_probs+1:
	    self.last_probs.popleft()
	if len(self.last_probs) == self.num_probs and sum([1 for x in self.last_probs if x < self.prob_thresh]) + 1>= self.num_probs and not self.reinitialized:
	    self.opt.reinitialize()
	    self.reinitialized = True
	    
	    return True
	    
	return False

# -----------------------------------------------------------------------------------------------

    def get_weights(self, t):
	"""
	For a future implementation using autocorrelations
	"""
	
	netdisp_command = self.spec.generate_net_display_command(t)
	process = subprocess.Popen(netdisp_command, shell=False, stdout=subprocess.PIPE)
	result = process.communicate()
	
	w_list = result[0].split('\n')
	for item in w_list:
	    print item
 
 # -----------------------------------------------------------------------------------------------
 
    def acceptence_rate(self):
	"""
	Get the latest acceptence rate
	"""
	
	accpt_command = self.spec.generate_accpt_command()
	process = subprocess.Popen(accpt_command, shell=False, stdout=subprocess.PIPE)
	result = process.communicate()
	
	w_list = result[0].strip('\n').split('\n')
	w_list = w_list[len(w_list) - self.step_size - 1:len(w_list)-1]
	w_list = [float(x.split()[1]) for x in w_list]
	
	return 1 - mean(w_list)
    
# -----------------------------------------------------------------------------------------------
    
    def annealing_schedule(self):
	return float(self.iter_ct+1.0)**-0.4

# -----------------------------------------------------------------------------------------------

    @staticmethod
    def sqrt_err(result):
	#splited = result.split('\n')
	for line in result:
	    if 'total' in line:
		splitted = line.strip('()').split()
		return float(0.01-float(splitted[len(splitted)-1].split('+-')[0]))*1000

# -----------------------------------------------------------------------------------------------

    @staticmethod
    def class_err(result):
	#splited = result.split('\n')
	for line in result:
	    if 'Fraction of guesses that were wrong' in line:
		splitted = line.split()
		return float(0.2 - float(splitted[len(splitted)-1].split('+-')[0]))*100

# -----------------------------------------------------------------------------------------------

    def setup_ceiling(self):
	self.step_size = int(floor(float(self.super_transition_steps)/self.spec.lf_step))
	self.spec.ceiling = self.spec.ceiling + self.step_size

# -----------------------------------------------------------------------------------------------

    def starter_run(self, logger):
	# Change stepsize and epsilon
	mcspec_command = self.spec.generate_mcspec_command()
	mcspec_command_str = netspec.to_string(mcspec_command)
	print mcspec_command_str
	#logger.info(mcspec_command_str)
	
	retcode = subprocess.check_call(mcspec_command)
	print "Starter Run: Finished setting specs."
	logger.info("Starter Run: Finished setting specs.")
	
	# Run the chain for a little bit
	netmc_command = self.spec.generate_netmc_command()	
	netmc_command_str = netspec.to_string(netmc_command)
	retcode = subprocess.check_call(netmc_command)
	print "Starter Run: Finished running the chain."
	logger.info("Starter Run: Finished running the chain.")

# -----------------------------------------------------------------------------------------------

    def opt_iter(self, logger):
	
	# Change stepsize and epsilon
	mcspec_command = self.spec.generate_mcspec_command()
	mcspec_command_str = netspec.to_string(mcspec_command)
	print mcspec_command_str
	
	retcode = subprocess.check_call(mcspec_command)
	print "	Finished setting specs."
	logger.info("	Finished setting specs.")
	
	# Run the chain for a little bit
	netmc_command = self.spec.generate_netmc_command()
	
	netmc_command_str = netspec.to_string(netmc_command)
	retcode = subprocess.check_call(netmc_command)
	print "	Finished running the chain."
	logger.info("	Finished running the chain.")
	
	# Run prediction and calculate reward
	netpred_command = self.spec.generate_netpred_command(self.spec.ceiling - self.step_size+1)
	netpred_command_str = netspec.to_string(netpred_command)
	
	process = subprocess.Popen(netpred_command, shell=False, stdout=subprocess.PIPE)
	result = process.communicate()
	
	if self.spec.model_spec == 'binary':
	    reward = facilities.class_err(result)
	elif self.spec.model_spec == 'real':
	    reward = facilities.sqrt_err(result)
	elif self.spec.model_spec == 'class':
	    reward = facilities.class_err(result)
	print "	Finished prediction."
	print "	Reward:", reward
	
	logger.info("	Finished prediction.")
	logger.info("	Reward: " + str(reward))
	
	# NOTE: Perform Bayesian optimization here
	self.bayesian_opt(reward, self.pure_bayes, logger)
	
	print "	New params:", self.spec.epsilon, self.spec.lf_step
	logger.info("	New params: " + str(self.spec.epsilon) + " " + str(self.spec.lf_step))

	self.setup_ceiling()
	
	print "	step_size:", self.step_size
	logger.info("	step_size: " + str(self.step_size))
	
	self.iter_ct = self.iter_ct + 1
	
# -----------------------------------------------------------------------------------------------

    def bayesian_opt(self, reward, pure_bayes, logger):
	
	if self.abandon_model:
	    # Get the extreme probability
	    extreme_prob = 0
	    if pure_bayes:
		extreme_prob = self.opt.prob_obs_x_or_extm([self.spec.epsilon, self.spec.lf_step], reward)[0]
	    else:
		extreme_prob = self.opt.prob_obs_x_or_extm([self.spec.epsilon*self.spec.lf_step], reward)[0]
		
	    print "	Extreme Prob:", extreme_prob
	    logger.info("	Extreme Prob: " + str(extreme_prob))
	    
	    # NOTE: extreme prob here
	    model_abandoned = self.update_last_probs(extreme_prob)
	    if model_abandoned:
		print "MODEL ABANDONED!."
		logger.info("MODEL ABANDONED!")
	
	
	# Update Model
	if pure_bayes:
	    self.opt.update([self.spec.epsilon, self.spec.lf_step], reward)
	else:
	    self.opt.update([self.spec.epsilon*self.spec.lf_step], reward)
	
	# Get the acceptence rate
	accpt_rate = self.acceptence_rate()

	print "	Finished Update."
	print "	Average accpt rate:", str(accpt_rate)
	logger.info("	Finished Update.")
	logger.info("	Average accpt rate: " + str(accpt_rate))
	
	# Do optimization
	x = self.opt.bf_opt(float(self.iter_ct+1))
	
	if pure_bayes:
	    self.spec.epsilon = x[0]
	    self.spec.lf_step = int(floor(x[1]))
	else:
	    print "	New Trajectory length:", x[0]
	    logger.info("	New Trajectory length: " + str(x[0]))
	    if accpt_rate > 0.7:
		self.spec.epsilon = self.spec.epsilon + self.annealing_schedule()*self.anneal_const
	    elif accpt_rate < 0.6:
		self.spec.epsilon = self.spec.epsilon - self.annealing_schedule()*self.anneal_const
	    self.spec.lf_step = int(float(x[0])/self.spec.epsilon)
