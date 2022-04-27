import os
import sys
from subprocess import run, PIPE, Popen, TimeoutExpired 
import re
import glob
import fileinput

pwd = run("pwd",shell=True,stdout=PIPE) 
pwd = pwd.stdout.decode('utf8').rstrip() #get the path of current running directory

def search_oom(slurm_path): #function to search oom error in slurm-number.out file
    with open(slurm_path,'r') as f:
        lines = f.readlines()
        flag = 0 
        for line in lines:
            if 'out-of-memory'in line or 'oom-kill event' in line:
                flag = 1
            else:
                flag = 0
    return flag


def get_results(result_path): # function to judge whether a system is converged or not 
    with open(result_path,'r') as f:
        lines = f.readlines()
        flag = 0
        line_num = 0
        flag2 = 0
        for line in reversed(lines):
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

            if flag == 5: # if all above five requirement are met, the system is converged and the flag will equal to 5
                break
            else:
                continue
    return flag
 
    
    
def change_nodes(sub_path): # if the error in slurm-number.out file is oom, change node from 4 to 6 in submission file
    with fileinput.input(files=(sub_path), inplace=True) as f:
        for line in f:
            if 'SBATCH --nodes' in line:
                line = line.replace('nodes=4','nodes=6')
            sys.stdout.write(line)

def cp2k_remove(cp2k_path): # if the system needs to restart, modify the cp2k.inp file
    i = 0
    with fileinput.input(files=(cp2k_path), inplace=True) as f:
        for line in f:
            if i < 3 and '!' in line:
                line = line.replace('!',' ')
                sys.stdout.write(line)
                i = i +1
            else:
                i = i +1
                sys.stdout.write(line)

def search_restart(slurm_path):
    temp = 0
    with open(slurm_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            if "MXene-RESTART.kp" in line:
                temp = 1
            else:
                continue
    return temp

def cp2k_restart(cp2k_path):
    with fileinput.input(files=(cp2k_path), inplace=True) as f:
        for line in f:
#           ''' if "MXene-RESTART.kp.bak-3" in line:
 #               sys.stdout.write(line)
  #              continue
#
 #           elif 'MXene-RESTART.kp' in line:
  #              line = line.replace('      WFN_RESTART_FILE_NAME  MXene-RESTART.kp','      WFN_RESTART_FILE_NAME  MXene-RESTART.kp.bak-3')
   #             sys.stdout.write(line)
    #        else:                
     #           sys.stdout.write(line)'''
#
            if "MXene-RESTART.kp" in line:
                pass
            else:
                sys.stdout.write(line)


all_restart = []
i = 0
all_running_path = os.path.join(pwd,'all_running_path')
with open(all_running_path,'r') as f:
    lines = f.readlines()
    temp_count = []
    flag_count = []
    for line in lines:
        if i in range(140):
            line = line.strip('\n')
            sub_path = os.path.join(line,'submissionfile')
            cp2k_path = os.path.join(line,'cp2k.inp')
            result_path = os.path.join(line,'RESULTS')
            result_flag = get_results(result_path) 
            if result_flag != 5:
                cp2k_remove(cp2k_path)
                all_restart.append(line)
                
                for f_name in os.listdir(line):
                   # print(f_name)
                    if f_name.startswith('slurm') and f_name.endswith('.out'):
                        slurm_path = os.path.join(line,f_name)
                        print(slurm_path)
                        temp = search_restart(slurm_path)
                        temp_count.append(temp)
                        flag = search_oom(slurm_path)
                        flag_count.append(flag)
                        #print(flag)
                            
                    else:
                        continue
                if sum(temp_count) >= 1:
                    cp2k_restart(cp2k_path)
                else:
                    pass
                if sum(flag_count) >= 1:
                    change_nodes(sub_path)
                else:
                    pass
            i = i+1
        else:
            i = i+1
            #print(i)

all_restart_path = os.path.join(pwd,'all_restart_path')
with open(all_restart_path,'w') as f:
    f.writelines("%s\n" % l for l in all_restart)
    
for i in range(len(all_restart)):
    proc = Popen(['sbatch','submissionfile'],cwd=all_restart[i], stdout=PIPE)
        
            
            
            
            
            
                
    
