import ConfigParser

class config:
    def __init__(self, config_file):
	self.config_file = config_file
    
    def get_run_counter(self):
	config = ConfigParser.RawConfigParser()
	config.read(self.config_file)
	return config.getint('Section3', 'cur_counter')
	
    def get_and_set_run_counter(self):
	config = ConfigParser.RawConfigParser()
	config.read(self.config_file)
	
	counter = config.getint('Section3', 'cur_counter')
	config.set('Section3', 'cur_counter', str(counter + 1))
	with open(self.config_file, 'wb') as configfile:
	    config.write(configfile)
	return counter

    def get_run_log_path(self, data_set_name):
	config = ConfigParser.RawConfigParser()
	config.read(self.config_file)

	run_log_path = config.get('Section4', 'MADELON')
	
	return run_log_path

    def get_command_path(self):
	config = ConfigParser.RawConfigParser()
	config.read(self.config_file)

	cmd_path = config.get('Section1', 'command_path')
	
	return cmd_path
	
    def get_file_path(self, name = '', cur_counter = ''):
	config = ConfigParser.RawConfigParser()
	config.read(self.config_file)
	file_path = config.get('Section1', 'file_path')
	return file_path + str(name) + str(cur_counter)
	
    def get_data_path(self, data_set_name):
	config = ConfigParser.RawConfigParser()
	config.read(self.config_file)
	data_path = config.get('Section2', data_set_name)
	
	return data_path
