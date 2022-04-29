from subprocess import run, PIPE, Popen, TimeoutExpired
import os
import re
from pathlib import Path
import pandas as pd


def get_results(result_path):
    with open(result_path,'r') as f:
        lines = f.readlines()
        flag = 0 
        line_num = 0
        flag2 = 0
        for i,line in enumerate(reversed(lines)):
            if 'GEOMETRY OPTIMIZATION COMPLETED' in line:
                flag = flag + 1
                flag2 = 1
           
            if flag2 == 1:
                if 'Convergence in step size   =                  YES' in line:
                    flag = flag + 1 
                if 'Convergence in RMS step    =                  YES' in line:
                    flag = flag + 1
                if 'Conv. in gradients         =                  YES' in line:
                    flag = flag + 1
                if 'Conv. in RMS gradients     =                  YES' in line:
                    flag = flag + 1
            else:
                continue
            print(flag)
            if flag == 5: 
                line_num = i
                #print('line num ={}'.format(line_num))
                break       
            else:
                continue

        energy = None
        if flag == 5:

            for j,line in enumerate(reversed(lines)):
                if j <= line_num:
                    if 'ENERGY| Total FORCE_EVAL' in line:
                        energy = line.rsplit(None, 1)[-1]
                        print('energy is{}'.format(energy))
                else:
                    break
        else:
            print('RESULTS file in invalid')
            pass
  
                        
    return energy
                    
pwd = run("pwd",shell=True,stdout=PIPE) 
pwd = pwd.stdout.decode('utf8').rstrip()
COOH_running_path = os.path.join(pwd,'COOH_running_path')
all_results = {}
with open(COOH_running_path,'r') as f:
    lines = f.readlines()
    i = 0

    for line in lines:
        #if i == 6:
         #   i = i+1
        #if i<140:
        line = line.strip('\n')
        p = Path(line)
        p = p.parts[7:]
        key_name = '-'.join(p) 
        print(line)
        result_path = os.path.join(line,'RESULTS')
        all_results.update({key_name:get_results(result_path)})
         #   i = i+1
       # else:
        #    break
print(all_results)            
pd.DataFrame.from_dict(data=all_results, orient='index').to_csv('COOH_results.csv', header=False)
