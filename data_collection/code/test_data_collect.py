import os
import random
import shutil
cat_dict = dict()
data_dir = '../data/train_data'
data_list = os.listdir(data_dir)


for data_name in data_list:
    cat = data_name.split('_')[1]
    
    if cat in list(cat_dict.keys()):
       
        cat_dict[cat] += 1
    else:
        cat_dict[cat] = 1
print(cat_dict)
        