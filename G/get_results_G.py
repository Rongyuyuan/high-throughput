from subprocess import run, PIPE, Popen, TimeoutExpired
import os
import re
from pathlib import Path
import pandas as pd

def get_results(Result_path):
    E_pot=None
    E_ZPE=None
    Cv_harm=None
    T_S=None
    F=None
    with open(Result_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            #print(line)
            if "E_pot" in line:
                E_pot = line.split()[-2]
            if "E_ZPE" in line:
                E_ZPE = line.split()[-2]
            if "Cv_harm" in line:
                Cv_harm = line.split()[-2]
            if "-T*S" in line:
                T_S = line.split()[-2]
            if "F" and "eV" in line: 
                F = line.split()[-2]
    return E_pot, E_ZPE, Cv_harm, T_S, F

pwd = run("pwd",shell=True,stdout=PIPE) 
pwd = pwd.stdout.decode('utf8').rstrip()
all_running_path = os.path.join(pwd,"results_path")
all_results = {}
with open(all_running_path,'r') as f:
    lines = f.readlines()
    i = 0
    for line in lines:
        line = line.strip('\n')
        p = Path(line)
        p = p.parts[8:]
        key_name = '-'.join(p) 
        result_path = os.path.join(line,'Results')
        if os.path.exists(result_path):
            filesize = os.path.getsize(result_path)
            if filesize != 0:
                print(result_path)
                all_results.update({key_name:get_results(result_path)})
header_name = ["E_pot","E_ZPE","Cv_harm","-T*S","F"]
pd.DataFrame.from_dict(data=all_results, orient='index').to_csv('deltaG_results.csv', header=header_name)
