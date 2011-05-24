import ConfigParser

config = ConfigParser.RawConfigParser()

config.add_section('Section1')
config.set('Section1', 'command_path', '/home/ziyuw/projects/AHMC/fbm.2004-11-10/')
config.set('Section1', 'file_path', '/home/ziyuw/projects/AHMC/log/')
config.add_section('Section2')
config.set('Section2', 'MADELON', '/home/ziyuw/projects/AHMC/Data/MADELON/MADELON/')
config.add_section('Section3')
config.set('Section3', 'cur_counter', '0')
config.add_section('Section4')
config.set('Section4', 'MADELON', '/home/ziyuw/projects/AHMC/run_log/')
# Writing our configuration file to 'path_config.cfg'
with open('path_config.cfg', 'wb') as configfile:
    config.write(configfile)

