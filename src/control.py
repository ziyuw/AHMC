import subprocess
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

class netspec:
    def __init__(self, file_path, command_path, data_file):
	self.file_path = file_path # The location that stores the log file
	self.command_path = command_path
	self.num_input_units = 0; self.input_offset = '-'
	self.num_hidden_layers = 0; self.hidden_layer_specs = []
	self.num_output_units = 0; self.hidden_output_weights = '-'
	self.input_output_weights = '-'; self.output_bias = '-'
	
	self.data_file = data_file; self.train_range = ''
	self.test_range = ''; self.model_spec = 'binary'
	
	self.init_value = 0.5; self.repeat_iteration = 10
	
	self.lf_step = 1000; self.window_size = 10
	self.epsilon = 0.25; self.ceiling = 100
	
	
    def make_string(self, cmd):
	for i in range(size(cmd)):
	    cmd[i] = str(cmd[i])
	return cmd
	
    def generate_netspec_command(self):
	cmd = []; cmd.append(self.command_path+'net/net-spec')
	cmd.append(self.file_path); cmd.append(self.num_input_units)
	for hl in self.hidden_layer_specs:
	    cmd.append(hl.num_units)
	cmd.append(self.num_output_units); cmd.append('/')
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
	# NOTE: HERE I ASSUME THE BIANRY CASE HOLDS
	cmd.append(2)
	cmd.append('/')
	cmd.append(self.data_file+'@'+self.train_range)
	cmd.append('.')
	cmd.append(self.data_file+'@'+self.test_range)
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
	"""
	cmd = []; cmd.append(self.command_path+'mc/mc-spec')
	cmd.append(self.file_path); cmd.append('repeat')
	cmd.append(self.repeat_iteration); cmd.append('sample-noise')
	cmd.append('heatbath'); cmd.append('hybrid')
	cmd.append(str(self.lf_step) + ":" + str(self.window_size))
	cmd.append(self.epsilon)
	
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
	cmd.append('tmp'); cmd.append(self.file_path)
	cmd.append(str(discard)+':');
	
	return self.make_string(cmd)
    
    @staticmethod
    def to_string(cmd):
	cmd_string = ''
	for i in range(size(cmd)):
	    cmd_string = cmd_string + cmd[i] + ' '
	return cmd_string
	
class facilities:
    
    def __init__(self, super_transition_steps, spec):
	self.super_transition_steps = super_transition_steps
	self.spec = spec
	self.spec.ceiling = int(floor(float(self.super_transition_steps)/self.spec.lf_step))
	self.step_size = self.spec.ceiling
	self.iter_ct = 0
    
    @staticmethod
    def class_err(result):
	#splited = result.split('\n')
	for line in result:
	    if 'Fraction of guesses that were wrong' in line:
		splitted = line.split()
		return 1 - float(splitted[len(splitted)-1].split('+-')[0])
		
    def opt_iter(self, opt, logger):
	
	# Change stepsize and epsilon
	mcspec_command = self.spec.generate_mcspec_command()
	mcspec_command_str = netspec.to_string(mcspec_command)
	#logger.info(mcspec_command_str)
	
	retcode = subprocess.check_call(mcspec_command)
	print "	Finished setting specs."
	logger.info("	Finished setting specs.")
	
	# Run the chain for a little bit
	netmc_command = self.spec.generate_netmc_command()
	
	netmc_command_str = netspec.to_string(netmc_command)
	#logger.info(netmc_command_str)
	
	retcode = subprocess.check_call(netmc_command)
	print "	Finished running the chain."
	logger.info("	Finished running the chain.")
	
	# Run prediction and calculate reward
	netpred_command = self.spec.generate_netpred_command(self.spec.ceiling - self.step_size+1)
	netpred_command_str = netspec.to_string(netpred_command)
	#logger.info(netpred_command_str)
	
	process = subprocess.Popen(netpred_command, shell=False, stdout=subprocess.PIPE)
	result = process.communicate()
	reward = facilities.class_err(result)
	print "	Finished prediction."
	print "	Reward:", reward
	
	logger.info("	Finished prediction.")
	logger.info("	Reward: " + str(reward))

	opt.update([self.spec.epsilon, self.spec.lf_step], reward)
	print "	Finished Update."
	
	logger.info("	Finished Update.")
	
	# Do optimization
	x = opt.bf_opt(float(self.iter_ct+1))
	
	self.spec.epsilon = x[0]
	self.spec.lf_step = int(floor(x[1]))
	
	print "	New params:", self.spec.epsilon, self.spec.lf_step
	logger.info("	New params: " + str(self.spec.epsilon) + " " + str(self.spec.lf_step))
	
	self.step_size = int(floor(float(self.super_transition_steps)/self.spec.lf_step))
	self.spec.ceiling = self.spec.ceiling + self.step_size
	print "	step_size:", self.step_size
	logger.info("	step_size: " + str(self.step_size))
	
	self.iter_ct = self.iter_ct + 1
