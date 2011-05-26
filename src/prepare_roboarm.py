import subprocess
from control import *
from config_setup import *
from config import *

conf = config('path_config.cfg')
path = conf.get_data_path('ROBOARM')
train_data = open(path + 'robot.data', 'r')
label_file = open(path + 'robot.targets', 'r')

final_file = open(path + 'combined_robot.data', 'w')


while True:
    train_line = train_data.readline()
    if train_line == '':
	break;
    label_line = label_file.readline()
    final_file.write(train_line.strip('\n') + ' ' + label_line)

train_data.close()
label_file.close()
final_file.close()