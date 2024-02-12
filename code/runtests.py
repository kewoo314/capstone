#!/usr/bin/env python3

import subprocess
import itertools
import sys

if len(sys.argv) < 6:
    print("Usage:",sys.argv[0],"groupsize splits repeats prefix infiledata*n");
    exit(-1)

groupsize = int(sys.argv[1])
splits = int(sys.argv[2])
repeats = int(sys.argv[3])
prefix = sys.argv[4]

if groupsize > 3:
    print("Too many in a group (<3)")
    exit(-1)

srcs = ["cn", "fn", "lv", "fp"]
trans = ["raw", "2gramI","3gramC","terms"]
mods = ["ratio2", "str", "calc"]

base = []
for s in srcs:
    for t in trans:
        for m in mods:
            base.append(s+"_"+t+"_"+m)

basedir = "/io1/home1/kewoo/forkaris/"

count = 0
print("src,trans,mod,f2,f1,prec,recall")
subcount=0
strcombo = f"{prefix}_{groupsize:02d}_{repeats:02d}x_{count:03d}"
oname = f"{basedir}logs/{strcombo}.out"
for combo in itertools.combinations(base,groupsize):
    #strcombo = "_".join(combo)
    strcombo = f"{prefix}_{groupsize:02d}_{repeats:02d}x_{count:03d}"
    oname = f"{basedir}logs/{strcombo}.out"
    if subcount == 0:
        qsub = open(f"{basedir}qsubs/{strcombo}.qsub","w")
        print("#!/bin/bash\n#PBS -l nodes=1\n#PBS -q nogpu """, file = qsub)  
    
    print(f"conda run -n tflow {basedir}code/calculatef2.py",splits, repeats,f"{oname}.predict",*sys.argv[5:],"-o",*combo,file=qsub)  

    subcount+=1
    if subcount == 10:
         count+=1
         qsub.close()
         subcount=0        
         runlist = ["qsub","-q","nogpu","-w","e","-N",strcombo,"-o",oname,"-e",oname+".err",f"{basedir}qsubs/{strcombo}.qsub"]
         #print(*runlist)
         subprocess.Popen(["qsub","-q","nogpu","-w","e","-N",strcombo,"-o",oname,"-e",oname+".err",f"{basedir}qsubs/{strcombo}.qsub"])
if subcount != 0:
    qsub.close()
    runlist = ["qsub","-q","nogpu","-w","e","-N",strcombo,"-o",oname,"-e",oname+".err",f"{basedir}qsubs/{strcombo}.qsub"]
    subprocess.Popen(["qsub","-q","nogpu","-w","e","-N",strcombo,"-o",oname,"-e",oname+".err",f"{basedir}qsubs/{strcombo}.qsub"])
     
#        with open(f"{basedir}qsubs/{strcombo}.qsub","w") as qsub:
#            oname = f"{basedir}logs/{strcombo}.out"
#           
#           
#            runlist = ["qsub","-q","gpu","-w","e","-N",strcombo,"-o",oname,"-e",oname+".err",f"{basedir}qsubs/{strcombo}.qsub"]
#            print(*runlist)
#            subprocess.Popen(["qsub","-q","gpu","-w","e","-N",strcombo,"-o",oname,"-e",oname+".err",f"{basedir}qsubs/{strcombo}.qsub"])
#        subprocess.run(["./calculatef2.py",*sys.argv[2:],"-o",*combo],stdout=output)
