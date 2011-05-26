
test_output = open('/home/ziyuw/projects/AHMC/result/valid.txt', 'r')
formatted_test_output = open('/home/ziyuw/projects/AHMC/result/madelon_valid.resu', 'w')

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

for line in test_output:
    if not line.strip() == '' and is_number(line.split()[1]):
	line_split = str(int(line.split()[1])*2-1) + '\n'
	formatted_test_output.write(line_split)
    
test_output.close()
formatted_test_output.close()