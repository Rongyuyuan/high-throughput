import os
from subprocess import run, PIPE, Popen, TimeoutExpired 

# when vibxxxxx.pckl for a job is empty after running, delete files in the directory and submit the job again
def del_empty(G_path):
    for f_name in os.listdir(G_path):
        if f_name.startswith('vib') and f_name.endswith('.pckl'):
            file_path = os.path.join(line,f_name)
            filesize = os.path.getsize(file_path)
            if filesize == 0:
                #os.remove(file_path)
                return file_path

pwd = run("pwd",shell=True,stdout=PIPE) 
pwd = pwd.stdout.decode('utf8').rstrip() #get the path of current running directory
all_running_path = os.path.join(pwd,'all_running_path')
all_del_file_path = []
with open(all_running_path,"r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        G_path = line
        if del_empty(G_path) != None:
            all_del_file_path.append(del_empty(G_path))
            os.remove(del_empty(G_path))

with open(os.path.join(pwd,"all_del_file_path"),'w') as f:
    for item in all_del_file_path:
        f.write("%s\n" % item)        
        
for i in range(len(all_del_file_path)):
    proc = Popen(['sbatch','submissionfile'],cwd=all_del_file_path[i], stdout=PIPE) 

