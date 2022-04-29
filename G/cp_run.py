import shutil, os
from subprocess import run, PIPE, Popen, TimeoutExpired 

pwd = run("pwd",shell=True,stdout=PIPE) 
pwd = pwd.stdout.decode('utf8').rstrip()

# copy run.py to sub-directories
tms = ["Ti3/Ni","Ti3/Pt","Mo2/Ru"]
intermediates = ["H2","CO2","OCHO","HCOOH","CHO","OCH2","OCH3","COOH","CHOH","CH2OH","CH2","CH3","O","CO"]
'''all_running_path = []
for tm in tms:
    for item in intermediates:
        dir_path = os.path.join(pwd,tm,item,"G")
        original_path = os.path.join(pwd,"run.py")
        all_running_path.append(dir_path)
        shutil.copy(original_path,dir_path)

run_file_path = os.path.join(pwd,"results_path")
with open(run_file_path,'w') as f:
    for item in all_running_path:
        f.write("%s\n" % item)
'''
all_run_path = []
with open('all_running_path','r') as f:
    lines = f.readlines()
    for item in lines:
        all_run_path.append(item.rstrip())

# run jobs
for i in range(len(all_run_path)):
    proc = Popen(['sbatch','submissionfile'],cwd=all_run_path[i], stdout=PIPE)                
