import os
import sys
from argparse import ArgumentParser

from datagen import DataGen

parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, help='data directory')
parser.add_argument('--data_fn', type=str, help='data file that indexs all shape-ids')
parser.add_argument('--primact_types', type=str, help='list all primacts [separated by comma], default: None, meaning all', default=None)
parser.add_argument('--num_processes', type=int, default=40, help='number of CPU cores to use')
parser.add_argument('--num_epochs', type=int, default=160, help='control the data amount')
parser.add_argument('--starting_epoch', type=int, default=0, help='help to resume. If previous generating does not generate the expected amount of data, when resuming, set this term to the previous epoch number to prevent from overlapping')
parser.add_argument('--ins_cnt_fn', type=str, help='a file listing all category instance count, which is used to balance the interaction data amount to make sure that all categories have roughly same amount of data interaction, regardless of different shape counts in these categories')
parser.add_argument('--mode', type=str, help='train or test; control the categories')
conf = parser.parse_args()



if conf.mode == 'train' and conf.primact_types == 'pulling':
    #set train categories
    conf.category_types = ['Safe', 'Door','Display','Refrigerator' ,'Laptop','Lighter','Microwave','Mouse','Box','TrashCan','KitchenPot','Suitcase','Pliers','StorageFurniture','Remote','Bottle'
    , 'FoldingChair','Toaster','Lamp','Dispenser', 'Cart', 'Globe','Eyeglasses','Pen','Switch','Printer','Keyboard','Fan','Knife','Dishwaher']
elif conf.mode == 'test' and conf.primact_types == 'pulling':
    #set test categories
    conf.category_types = ['Safe', 'Door','Display','Refrigerator' ,'Laptop','Lighter','Microwave','Mouse','Box','TrashCan','KitchenPot','Suitcase','Pliers','StorageFurniture','Remote','Bottle'
    , 'FoldingChair','Toaster','Lamp','Dispenser', 'Cart', 'Globe','Eyeglasses','Pen','Switch','Printer','Keyboard','Fan','Knife','Dishwaher','Toilet', 'Scissors','Table', 'Stapler','USB',
    'WashingMachine', 'Oven','Faucet', 'Phone','Kettle','Window']

hard_train_cat = ['Dispenser','Globe','Remote','Cart','Fan','Knife']
easy_train_cat = ['StorageFurniture','Pen','Laptop','Microwave','Refrigerator','Safe']

cat2freq = dict()
with open(conf.ins_cnt_fn, 'r') as fin:
    for l in fin.readlines():
        cat, _, freq = l.rstrip().split()
        #hard categories are harder to collect success samples, therefore, increase the frequency of interacting with these categories to keep the category balance
        if cat in hard_train_cat: 
            freq *= 2
            cat2freq[cat] = freq
        elif cat in easy_train_cat:
            freq = int(float(freq) / 1.2)
            cat2freq[cat] = freq
        cat2freq[cat] = int(freq)

datagen = DataGen(conf.num_processes)
primact_type = conf.primact_types
with open(conf.data_fn, 'r') as fin:
    for l in fin.readlines():
        shape_id, cat = l.rstrip().split()
        if cat in conf.category_types:
            for epoch in range(conf.starting_epoch, conf.starting_epoch+conf.num_epochs):
                for cnt_id in range(cat2freq[cat]):
                    datagen.add_one_collect_job(conf.data_dir, shape_id, cat, cnt_id, primact_type, epoch)

datagen.start_all()

print('start generating data')
