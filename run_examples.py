import os
import sys
import json
from multiprocessing import Pool
import tempfile
import subprocess

inputdir = "inputs/"
outputdir  = "example_outputs/"
for root, subdirs, files in os.walk(inputdir):
    ofile_root = root.replace(inputdir, outputdir)
    #os.makedirs(ofile_root, exist_ok=True)

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
            print(json.dumps(ifjson))
            
            
