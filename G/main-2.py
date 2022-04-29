import csv
import argparse
from subprocess import run, PIPE, Popen, TimeoutExpired 
import os
import shutil
import re
import sys
import itertools
import fileinput
import ase.io
from ase import Atoms
from ase.io import read, write


def make_input_file(pwd,tm,item):
    
    Ni_opt_path = os.path.join(pwd,tm,item,"MXene-pos-1.xyz")
    new_reverse_path = os.path.join(pwd,tm,item,"G","reverse_input_file.xyz")
    new_path = os.path.join(pwd,tm,item,"G","input_file.xyz")
    opt_xyz_file = open(Ni_opt_path,"r")
    lines = opt_xyz_file.readlines()
    reverse_input_file = open(new_reverse_path,"w")
    flag = 1
    for line in reversed(lines): #read MXene-pos-1.xyz file reversed and extract the last E
        if flag == 2:
            reverse_input_file.write(line)
            break
        else:   
            if 'E' not in line:
                reverse_input_file.write(line)
                flag = 1
            else:
                reverse_input_file.write(line)
                flag = 2
       
    
    opt_xyz_file.close() #close files that are opened
    reverse_input_file.close()
    
    with open(new_reverse_path,'r') as f:
        reverse_input_lines = f.readlines()
        with open(new_path,'w') as ipf:
            for line in reversed(reverse_input_lines):
                ipf.write(line)
                
    os.remove(new_reverse_path)
    return new_path

def reverse_TM(pwd,tm,item): #make input.xyz (put this in front of the file)
    input_path = os.path.join(pwd,tm,item,"G","input_file.xyz")
    output_path = os.path.join(pwd,tm,item,"G","input.xyz")
    structure = ase.io.read(input_path)
    struc_name = structure.get_chemical_symbols()
    with open(input_path,"r") as inp:
        lines = inp.readlines() 
        #print(lines[47])
        #print(struc_name[45])
        with open(output_path,"w") as f:
            for i in range(2):
                f.write(lines[i])
            if struc_name[45] != "Ru":
                for i in range(66,len(struc_name)+2):
                    f.write(lines[i])
                f.write(lines[65])
                for i in range(2,65):
                    f.write(lines[i])
                    
            else:
                for i in range(48,len(struc_name)+2):
                    f.write(lines[i])
                f.write(lines[47])
                for i in range(2,47):
                    f.write(lines[i])
                    
def make_sub_file(pwd,tm,item): #make submissionfile (copy from H2 which is converged)
    original_sub_path = os.path.join(pwd,"Ti3/Pt","OCH3","G","submissionfile")
    new_sub_path = os.path.join(pwd,tm,item,"G","submissionfile")
    with open(original_sub_path,'r') as of:
        lines = of.readlines()
        with open(new_sub_path,'w') as inf:
            for line in lines:
                if 'job-name' in line:
                    new_line = "#SBATCH --job-name={}_{}_{}_G\n".format(tm.split("/")[0],tm.split("/")[1],item)
                    inf.write(new_line)
                else:
                    inf.write(line)                 
                    
#Find the present working directory and store as a string
pwd = run("pwd",shell=True,stdout=PIPE) 
pwd = pwd.stdout.decode('utf8').rstrip()

tms =["Ti3/Ni","Ti3/Pt","Mo2/Ru"]
intermediates = ["OCHO","HCOOH","CHO","OCH2","OCH3","COOH","CHOH","CH2OH","CH2","CH3","O","CO"]

for tm in tms:
    for item in intermediates:
        dir_path = os.path.join(pwd,tm,item,"G")
        if not os.path.exists(dir_path): 
            os.makedirs(dir_path)
        make_input_file(pwd,tm,item)
        reverse_TM(pwd,tm,item)
        make_sub_file(pwd,tm,item)

