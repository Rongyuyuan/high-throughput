import os
from subprocess import run, PIPE, Popen, TimeoutExpired
from fnmatch import fnmatch

pwd = run("pwd",shell=True,stdout=PIPE) 
pwd = pwd.stdout.decode('utf8').rstrip()
print(pwd)

layers = ['Ti2','Ti3','Mo2','Mo3']
transition_metal = ["Co","Cu","Fe","Pt","Ru","Ni"]
intermediates = ["H2","CO2","OCHO","HCOOH","CHO","OCH2","OCH3"]

'''pattern2 = intermediates
pattern1 = "Ni"'''
all_run_path = []

for layer in layers:
    for tm in transition_metal:
       # i = 0
        for interm in intermediates:
           # if i <= 5:
            run_path = os.path.join(pwd,layer,tm,interm)
            all_run_path.append(run_path)
              #i += 1
            #else:
             #   continue
with open('all_running_path','w') as f:
    for item in all_run_path:
        f.write("%s\n" % item)
            
                
#print(all_run_path)
#for i in range(110,141,1):
#proc = Popen(['sbatch','submissionfile'],cwd=all_run_path[6], stdout=PIPE) 
