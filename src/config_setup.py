import ConfigParser

config = ConfigParser.RawConfigParser()

home_path = '/home/ziyuw/projects/AHMC/'

config.add_section('Section1')
config.set('Section1', 'command_path', home_path+'fbm.2004-11-10/')
config.set('Section1', 'file_path', home_path+'log/')
config.add_section('Section2')
config.set('Section2', 'MADELON', home_path+'Data/MADELON/MADELON/')
config.set('Section2', 'ROBOARM', home_path+'Data/RobotArm/')
config.add_section('Section3')
config.set('Section3', 'cur_counter', '0')
config.add_section('Section4')
config.set('Section4', 'MADELON', home_path+'run_log/')
config.set('Section4', 'ROBOARM', home_path+'run_log/')
# Writing our configuration file to 'path_config.cfg'
with open('path_config.cfg', 'wb') as configfile:
    config.write(configfile)

