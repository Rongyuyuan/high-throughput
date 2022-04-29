import shutil, os
from subprocess import run, PIPE, Popen, TimeoutExpired 

pwd = run("pwd",shell=True,stdout=PIPE) 
pwd = pwd.stdout.decode('utf8').rstrip()

tms = ["Ti3/Ni","Ti3/Pt","Mo2/Ru"]
intermediates = ["OCHO","HCOOH","CHO","OCH2","OCH3","COOH","CHOH","CH2OH","CH2","CH3","O","CO"]


MXene_pos_final_path = []
for tm in tms:
    for item in intermediates:
        rongyu_path = "/home/rongyu/projects/def-alsei/rongyu/MXene/Adsorption/{}/{}".format(tm,item)
        dir_path = os.path.join(pwd,tm,item,"G")
        original_path = os.path.join(rongyu_path,"MXene-pos-1.xyz")
        MXene_pos_final_path.append(dir_path)
        shutil.copy(original_path,dir_path)

run_file_path = os.path.join(pwd,"MXene_pos_final_path")
with open(run_file_path,'w') as f:
    for item in MXene_pos_final_path:
        f.write("%s\n" % item)