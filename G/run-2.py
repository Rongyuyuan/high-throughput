#imports
import numpy as np 
from ase.build import molecule
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo
from ase.io import read
from ase import io
from ase.calculators.emt import EMT
from ase.calculators.cp2k import CP2K
from pathlib import Path
import os
import re

def cp2kinp(temp,tm,item):
    temp = re.sub('[0-9]+', '', temp)
    if "H" not in list(item):
        return '''
    !&EXT_RESTART
     !RESTART_FILE_NAME MXene-1.restart
    !&END

    &GLOBAL
       RUN_TYPE GEO_OPT
       EXTENDED_FFT_LENGTHS .TRUE.
       FFTW_PLAN_TYPE ESTIMATE
       PREFERRED_FFT_LIBRARY FFTW3

    &END GLOBAL

    &MOTION
       &GEO_OPT
          MAX_ITER 400
          OPTIMIZER BFGS
       &END GEO_OPT
       &PRINT
          &TRAJECTORY
             FORMAT XYZ
          &END TRAJECTORY
       &END PRINT

    &END MOTION
    &FORCE_EVAL




       &DFT
          WFN_RESTART_FILE_NAME  MXene-RESTART.kp
          &MGRID
             NGRIDS 5
             RELATIVE_CUTOFF 40
          &END MGRID
          &QS
             EPS_DEFAULT 1.0E-12
             METHOD GPW
             EXTRAPOLATION USE_GUESS
          &END QS
          &SCF
             EPS_SCF 1.0E-6
             SCF_GUESS RESTART
             CHOLESKY  OFF
            ADDED_MOS  200
             &OUTER_SCF F
                EPS_SCF 1.0E-6
                MAX_SCF 20
             &END OUTER_SCF
             &SMEAR T
                METHOD  FERMI_DIRAC
                ELECTRONIC_TEMPERATURE     3.0000000000000000E+02
             &END SMEAR
             &MIXING T
                METHOD  BROYDEN_MIXING
                ALPHA     4.0000000000000002E-01
                BETA     1.5000000000000000E+00
                NMIXING  5
                NBUFFER  8
            &END MIXING
          &END SCF
          &XC
          &VDW_POTENTIAL                            ! ... dispersion interactions
             POTENTIAL_TYPE PAIR_POTENTIAL
             &PAIR_POTENTIAL
                TYPE DFTD3                          ! computed with the DFTD3 method
                REFERENCE_FUNCTIONAL PBE
                ! that requires the following parameters (in external file, specified here)
                PARAMETER_FILE_NAME ./dftd3.dat
                R_CUTOFF 15                         ! cutoff raddius for the dispersion interactions
             &END PAIR_POTENTIAL
          &END
    &END XC
          &KPOINTS
             SCHEME  MONKHORST-PACK  4 4 1
             FULL_GRID  .TRUE.
          &END KPOINTS



       &END DFT
       &SUBSYS

    &KIND {}
             BASIS_SET DZVP-MOLOPT-SR-GTH
          &END KIND
             &KIND C
             BASIS_SET DZVP-MOLOPT-SR-GTH
          &END KIND
             &KIND O
             BASIS_SET DZVP-MOLOPT-SR-GTH
           &END KIND
         &KIND {}
             BASIS_SET DZVP-MOLOPT-SR-GTH
          &END KIND

    &END SUBSYS
    &END FORCE_EVAL

    '''.format(temp,tm)
    else:
        return '''
    !&EXT_RESTART
     !RESTART_FILE_NAME MXene-1.restart
    !&END

    &GLOBAL
       RUN_TYPE GEO_OPT
       EXTENDED_FFT_LENGTHS .TRUE.
       FFTW_PLAN_TYPE ESTIMATE
       PREFERRED_FFT_LIBRARY FFTW3

    &END GLOBAL

    &MOTION
       &GEO_OPT
          MAX_ITER 400
          OPTIMIZER BFGS
       &END GEO_OPT
       &PRINT
          &TRAJECTORY
             FORMAT XYZ
          &END TRAJECTORY
       &END PRINT

    &END MOTION
    &FORCE_EVAL




       &DFT
          WFN_RESTART_FILE_NAME  MXene-RESTART.kp
          &MGRID
             NGRIDS 5
             RELATIVE_CUTOFF 40
          &END MGRID
          &QS
             EPS_DEFAULT 1.0E-12
             METHOD GPW
             EXTRAPOLATION USE_GUESS
          &END QS
          &SCF
             EPS_SCF 1.0E-6
             SCF_GUESS RESTART
             CHOLESKY  OFF
            ADDED_MOS  200
             &OUTER_SCF F
                EPS_SCF 1.0E-6
                MAX_SCF 20
             &END OUTER_SCF
             &SMEAR T
                METHOD  FERMI_DIRAC
                ELECTRONIC_TEMPERATURE     3.0000000000000000E+02
             &END SMEAR
             &MIXING T
                METHOD  BROYDEN_MIXING
                ALPHA     4.0000000000000002E-01
                BETA     1.5000000000000000E+00
                NMIXING  5
                NBUFFER  8
            &END MIXING
          &END SCF
          &XC
          &VDW_POTENTIAL                            ! ... dispersion interactions
             POTENTIAL_TYPE PAIR_POTENTIAL
             &PAIR_POTENTIAL
                TYPE DFTD3                          ! computed with the DFTD3 method
                REFERENCE_FUNCTIONAL PBE
                ! that requires the following parameters (in external file, specified here)
                PARAMETER_FILE_NAME ./dftd3.dat
                R_CUTOFF 15                         ! cutoff raddius for the dispersion interactions
             &END PAIR_POTENTIAL
          &END
    &END XC
          &KPOINTS
             SCHEME  MONKHORST-PACK  4 4 1
             FULL_GRID  .TRUE.
          &END KPOINTS



       &END DFT
       &SUBSYS

    &KIND {}
             BASIS_SET DZVP-MOLOPT-SR-GTH
          &END KIND
             &KIND C
             BASIS_SET DZVP-MOLOPT-SR-GTH
          &END KIND
             &KIND O
             BASIS_SET DZVP-MOLOPT-SR-GTH
           &END KIND
         &KIND {}
             BASIS_SET DZVP-MOLOPT-SR-GTH
          &END KIND
         &KIND H
             BASIS_SET DZVP-MOLOPT-SR-GTH
         &END KIND

    &END SUBSYS
    &END FORCE_EVAL

    '''.format(temp,tm) 

def ABC(temp):
    if temp == "Ti3":
        return [9.231 ,9.231 ,20,90,90,120]
    if temp == "Mo2":
        return [9.162 ,9.162 ,20,90,90,120]

def atom_num(item):
    if len(re.findall(r'\d+', item)) == 0:
        m = re.sub('[0-9]+', '', item)
        return len(list(m))
    else:
        n = 0
        m = re.sub('[0-9]+', '', item)
        print(len(list(item)))
        for num in re.findall(r'\d+', item):
            i = int(num)-1
            n += i
            print(n)
        return len(list(m))+n    
    
def get_EDFT(fp_pwd):
    input_path = os.path.join(fp_pwd,"input.xyz")
    with open(input_path,"r") as f:
        lines = f.readlines()
        return float(lines[1].split(" ")[-1])
    
file_path = os.path.realpath(__file__)
fp_pwd =  "/".join(list(file_path.split('/')[0:-1])) #file_path includes the file name (different from pwd path)
p = Path(file_path)
temp = p.parts[-5]
tm = p.parts[-4]
item = p.parts[-3]
inp = cp2kinp(temp,tm,item)

Ry=13.6056980659                    #Conversion Constant to eV
Hartree_to_eV = 27.211324570273

slab=io.read('input.xyz')              #optimized structure from cp2k-pos.xyz
slab.set_cell(ABC(temp))
slab.set_pbc(True)

EDFT = get_EDFT(fp_pwd)*Hartree_to_eV    #Input: DFT energy in eV 
#slab.calc = EMT()
slab.set_calculator(CP2K(cutoff=550*Ry,max_scf=140, uks=True,xc='PBE', pseudo_potential='GTH-PBE',inp=inp))


indices = []                    	 
for i in range(0,atom_num(item)):    #INPUT : specify which are adsorbate atoms by index number.Put the adsorbate all the way at the top of xyz file! This is for O
    indices.append(i)

#this section calculates the vibrational energy
vib = Vibrations(slab,indices=indices,name='vib')
vib.run()
vib.summary(log='vib.txt')

for mode in range(len(indices)*3):
                vib.write_mode(mode)


vib_energies = vib.get_energies()

print(vib_energies) #This prints the vibrational energies, make sure there are no imaginary numbers!

#this section inputs the necessary input to calculate thermodynamic properties

thermo = HarmonicThermo(vib_energies[:],  potentialenergy=EDFT )  #Imaginary numbers need to be left out! This is why there is "[3:]"!
G = thermo.get_helmholtz_energy(temperature=298.15)          #Input : Temperature
