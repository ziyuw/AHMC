import subprocess
from control import *
from config_setup import *

path = get_data_path('MADELON')
train_data = open(path + 'train.data.sel', 'r')
label_file = open(path + 'madelon_train.labels', 'r')

final_file = open(path + 'combined.data.sel', 'w')


while True:
    train_line = train_data.readline()
    if train_line == '':
	break;
    label_line = str((int(label_file.readline().strip('\n'))+1)/2)
    final_file.write(train_line.strip('\n') + ' ' + label_line + "\n")

train_data.close()
label_file.close()
final_file.close()