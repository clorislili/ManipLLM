import numpy as np
import json
from argparse import ArgumentParser
import utils
import os
def calculate_succ_ration(data_list_for_cat,conf,out_dir):
    out_info={}
    for cat in conf.category_types:
        if cat in data_list_for_cat.keys():
            succ_ration_list=[]
            for i in data_list_for_cat[cat]:
                try:
                    with open(os.path.join(i, 'result2.json'), 'r') as fin:
                        result_data = json.load(fin)
                        succ_ration_list.append(result_data['mani_succ'])
                except:
                    continue
           
            succ_ration_list = np.array(succ_ration_list)
            out_info['number_of_%s'%cat]= len(succ_ration_list)
            mean_value = np.mean(succ_ration_list.astype(float))
            out_info['mani_succ_ration_for_%s'%cat]= mean_value
        else:
            # print("there is no '%s' data "% cat)
            continue
    train_cat = ['Safe', 'Door','Display','Refrigerator' ,'Laptop','Lighter','Microwave','Mouse','Box','TrashCan','KitchenPot','Suitcase','Pliers','StorageFurniture','Remote','Bottle'
    , 'FoldingChair','Toaster','Lamp','Dispenser','Eyeglasses','Pen','Printer','Keyboard','Fan','Knife','Dishwaher']

    count_train = 0
    count_test = 0
    osum_train = 0
    osum_test = 0
    print(out_info)
    for i in range(0,len(out_info.keys()),2):
        if list(out_info.keys())[i].split('_')[-1] in train_cat:
            if 0.0 <= out_info[list(out_info.keys())[i+1]] and  out_info[list(out_info.keys())[i+1]] <= 1.0:
                osum_train += out_info[list(out_info.keys())[i]] * out_info[list(out_info.keys())[i+1]]
                # print(out_info[list(out_info.keys())[i]],out_info[list(out_info.keys())[i+1]])
                count_train += out_info[list(out_info.keys())[i]]
        else:
            if 0.0 <= out_info[list(out_info.keys())[i+1]] and  out_info[list(out_info.keys())[i+1]] <= 1.0:
                osum_test += out_info[list(out_info.keys())[i]] * out_info[list(out_info.keys())[i+1]]
                count_test += out_info[list(out_info.keys())[i]]

    print(f'test seen acc on {count_train} samples is {osum_train/count_train}, test unseen acc on {count_test} samples is {osum_test/count_test}')
    with open(os.path.join(out_dir, 'mani_succ_ration_for_cats.json'), 'w') as fout:
        json.dump(out_info, fout)
    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--primact_type', type=str, help='primact_type:pushing,pulling,pushing left,pulling left')
    parser.add_argument('--data_dir', type=str, help='data_dir for whole test data')
    parser.add_argument('--out_dir', type=str, help='out_dir for calculate_info')
    conf = parser.parse_args()
    

    conf.category_types = ['Safe', 'Door','Display','Refrigerator' ,'Laptop','Lighter','Microwave','Mouse','Box','TrashCan','KitchenPot','Suitcase','Pliers','StorageFurniture','Remote','Bottle'
    , 'FoldingChair','Toaster','Lamp','Dispenser','Toilet', 'Scissors','Table','USB',
    'WashingMachine', 'Oven','Faucet']
    conf.out_dir = os.path.join(conf.data_dir,'calculate_info')
    if not os.path.exists(conf.out_dir):
        os.makedirs(conf.out_dir)

    data_list_for_cat={}
    record_names = os.listdir(conf.data_dir)
   
    for record_name in record_names:
       
        if '.png' in record_name or '.json' in record_name:
            continue
        else:
           
            category= record_name.rstrip().split('_')[1]
            data_list_for_cat.setdefault(category,[]).append(os.path.join(conf.data_dir, record_name.rstrip()))
   
    calculate_succ_ration(data_list_for_cat,conf,conf.out_dir)
