import ConfigParser

config = ConfigParser.RawConfigParser()

config.add_section('Section1')
config.set('Section1', 'command_path', '/ubc/cs/home/z/ziyuw/projects/AHMC/fbm.2004-11-10/')
config.set('Section1', 'file_path', '/ubc/cs/home/z/ziyuw/projects/AHMC/log/')
config.add_section('Section2')
config.set('Section2', 'MADELON', '/ubc/cs/home/z/ziyuw/projects/AHMC/Data/MADELON/')

# Writing our configuration file to 'path_config.cfg'
with open('path_config.cfg', 'wb') as configfile:
    config.write(configfile)
    

def get_command_path():
    config = ConfigParser.RawConfigParser()
    config.read('path_config.cfg')

    cmd_path = config.get('Section1', 'command_path')
    
    return cmd_path
    
def get_file_path():
    config = ConfigParser.RawConfigParser()
    config.read('path_config.cfg')
    file_path = config.get('Section1', 'file_path')
    return file_path
    
def get_data_path(data_set_name):
    config = ConfigParser.RawConfigParser()
    config.read('path_config.cfg')
    data_path = config.get('Section2', data_set_name)
    
    return data_path
