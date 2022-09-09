import os
import sys
import json
from multiprocessing import Pool
import tempfile
import subprocess

import multiprocessing
import subprocess 
import shlex
import os
import numpy as np

from multiprocessing.pool import ThreadPool


def call_proc(cmd, ijson, id):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = '1'
    env["MKL_NUM_THREADS"] = '1'
    
    fname = "temp/infiles_"+str(id)
    errname = "temp/errfiles_"+str(id)
    f = open(errname, 'w')
    with open(fname, 'w') as of:
        json.dump(ijson, of)
    
    cmd = cmd + fname
    print(cmd)
    p = subprocess.Popen(shlex.split(cmd), stdout = subprocess.DEVNULL, stderr = f, env=env)
    f.close()
    out, err = p.communicate()
    if(os.path.exists(fname)):
        os.remove(fname)
    return (out, err)

pool = ThreadPool(16)
results = []


inputdir = "inputs/"
outputdir  = "example_outputs/"
counter = 0
for root, subdirs, files in os.walk(inputdir):
    ofile_root = root.replace(inputdir, outputdir)
    os.makedirs(ofile_root, exist_ok=True)

    for filename in files:
        if not "commented" in filename and not ".swp" in filename and not ".swo" in filename:
            ifile_path = os.path.join(root, filename)
            ofile_path = os.path.join(ofile_root, filename.replace(".in", ".out"))
            print('list_file_path = ' + ifile_path)
            print('list_file_path = ' + ofile_path)

            #with open(ifile_path, 'r') as f:
            #    print(f.read())
            f = open(ifile_path)
            ifjson = json.load(f)
            f.close()
            ifjson['output file'] = str(ofile_path)
            counter = counter+1
            results.append(pool.apply_async(call_proc, ("time bin/mlmctdh.x " , ifjson, counter)))
pool.close()
pool.join()

for result in results:
    out, err = result.get()

