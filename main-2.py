import csv
import argparse
from subprocess import run, PIPE, Popen, TimeoutExpired 
import os
import shutil
import re
import sys
import itertools
import ase.io
from ase import Atoms
from ase.io import read, write
import fileinput



# get input_file.xyz for other structures from template structure   
def make_input_file(pwd,layer,transition_metal,intermediates):
    
    Ni_opt_path = os.path.join(pwd,layer,'Ni',intermediates,'MXene-pos-1.xyz')
    new_reverse_path = os.path.join(pwd,layer,transition_metal,intermediates,"reverse_input_file.xyz")
    new_path = os.path.join(pwd,layer,transition_metal,intermediates,'input_file.xyz')
    opt_xyz_file = open(Ni_opt_path,"r")
    lines = opt_xyz_file.readlines()
    reverse_input_file = open(new_reverse_path,"w")
    flag = 1
    for line in reversed(lines): 

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
       
    
    opt_xyz_file.close()
    reverse_input_file.close()
    
    with open(new_reverse_path,'r') as f:
        reverse_input_lines = f.readlines()
        with open(new_path,'w') as ipf:
            for line in reversed(reverse_input_lines):
                #print(line)
                ipf.write(line)
   
    os.remove(new_reverse_path)
    return new_path

# use ase to change atoms in input_file.xyz 
def change_atom(input_path,output_path,layer,transition_metal):
    layer1 =['Mo2','Ti2']
    layer2 = ['Mo3','Ti3']
    #atoms=read(input_path)
    #atoms.center
    #atoms.set_cell([9.195,9.195,13.735,90,90,120])
    #atoms.set_pbc([1,1,0])


    #write(output_path,atoms)
    #write('post.cif',atoms)
    structure=ase.io.read(input_path)
    if layer in layer1:
        structure[45].symbol=transition_metal
        ase.io.write(output_path,structure)
    else:
        structure[63].symbol=transition_metal
        ase.io.write(output_path,structure)
        
# if there is no H element in the intermediate, delete H in cp2k.inp
# change transition metal and template atom
def cp2k_del_H(pwd,layer,transition_metal,intermediates):
    cp2k_file_path = os.path.join(pwd,layer,"Ni","CO2",'cp2k.inp')
    new_cp2k_path = os.path.join(pwd,layer,transition_metal,intermediates,'cp2k.inp')
    layer1 =['Ti3','Ti2']
    layer2 = ['Mo3','Mo2']
    results = ''.join([i for i in layer if not i.isdigit()])
    
    with open(cp2k_file_path,'r') as f:
        lines = f.readlines()
        with open(new_cp2k_path,'w') as nf: 
            nf.seek(0)
            #print(intermediates.find('H'))
            if intermediates.find('H') != -1:
                for i,line in enumerate(lines):
                    nf.write(line)

            else:
                for i,line in enumerate(lines):
                    if i not in [115,116,117,118]:
                        nf.write(line)
                    else:
                        continue
            nf.truncate()
            
    with fileinput.input(files=(new_cp2k_path), inplace=True) as f:
        
        for line in f:
            if '&KIND Ni' in line:
                line = line.replace('&KIND Ni','&KIND {}'.format(transition_metal))
            if '&KIND Ti' in line:
                if layer in layer1:
                    pass
                else:
                    line = line.replace('&KIND Ti','&KIND {}'.format(results))
            sys.stdout.write(line)    


# change submission file name
def make_sub_file(pwd,layer,transition_metal,intermediates):
    original_sub_path = os.path.join(pwd,"main_submissionfile")
    new_sub_path = os.path.join(pwd,layer,transition_metal,intermediates,"submissionfile")
    with open(original_sub_path,'r') as of:
        lines = of.readlines()
        with open(new_sub_path,'w') as inf:
            for line in lines:
                if 'job-name' in line:
                    new_line = "#SBATCH --job-name={}_{}_{}\n".format(layer,transition_metal,intermediates)
                    inf.write(new_line)
                else:
                    inf.write(line)


#Find the present working directory and store as a string
pwd = run("pwd",shell=True,stdout=PIPE) 
pwd = pwd.stdout.decode('utf8').rstrip()


layer = ["Ti2","Ti3","Mo2","Mo3"]
transition_metal = ["Co","Cu","Fe","Pt","Ru"]
intermediates = ["H2","CO2","OCHO","HCOOH","CHO","OCH2","OCH3"]

dir_list = []
tree_list = []
tree_list.append(layer)
tree_list.append(transition_metal)
tree_list.append(intermediates)
p = itertools.product(*tree_list)

# get all structures that need to run 
with open("input_file.csv",'w') as f:
    writer = csv.writer(f, delimiter=',')
    for item in p:
        #print(item)
        writer.writerow(item)


input_file = open("input_file.csv")
reader = csv.reader(input_file)
ignore_list = []


for i,row in enumerate(reader):
    #if i >= 100:
        #print(type(row))
        layer =row[0]
        transition_metal=row[1]
        intermediates=row[2]
        

        dir_path = os.path.join(pwd,layer,transition_metal,intermediates)

        if not os.path.exists(dir_path): 
            os.makedirs(dir_path)
        else:
            pass

        if intermediates in ignore_list:
            pass
        else:
            input_path = make_input_file(pwd,layer,transition_metal,intermediates)
            output_path = os.path.join(pwd,layer,transition_metal,intermediates,'input.xyz')
            change_atom(input_path,output_path,layer,transition_metal)
            make_sub_file(pwd,layer,transition_metal,intermediates)
            cp2k_del_H(pwd,layer,transition_metal,intermediates)
    #else:
        #continue

    





