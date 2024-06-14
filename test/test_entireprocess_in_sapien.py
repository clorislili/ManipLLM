import os
import sys
from argparse import ArgumentParser

from processgen import Processgen
import json
parser = ArgumentParser()
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--data_dir', type=str, help='data directory')
parser.add_argument('--num_processes', type=int, default=40, help='number of CPU cores to use')
parser.add_argument('--out_dir', type=str, help='outdata directory')
parser.add_argument('--use_mask', type=str, default=False, help='whether use movable mask')
conf = parser.parse_args()

if os.path.exists(conf.out_dir):
    pass
else:
    print('NO infer directory')
    exit()
    
processgen = Processgen(conf.num_processes)
record_names = os.listdir(conf.out_dir)
for record_name in record_names:
    processgen.add_one_test_job(record_name,conf)
processgen.start_all()
data_tuple_list = processgen.join_all()


