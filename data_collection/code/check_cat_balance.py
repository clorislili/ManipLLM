import os
import numpy as np

data_dir = '../data/train_data'
data_list = os.listdir(data_dir)
cat_cal = dict()
for data_name in data_list:
    cat = data_name.split('_')[1]
    if cat not in list(cat_cal.keys()):
        cat_cal[cat] =  1
    else:
        cat_cal[cat] +=  1
print(cat_cal)