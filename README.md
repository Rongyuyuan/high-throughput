# high-throughput
high-throughput code for running DFT calculation
```
main_submissionfile and cp2k.inp are shell scripts
all .py files, main_submissionfile and cp2k.inp should be in the same location


DFT running process:
1. main.py :create input.xyz for different structures from the template
2. run_MXene.py :run the jobs
3. COOH_restart.py or restart.py (if MXene-RESTART.kp error )
4. get_results_dft.py
```

