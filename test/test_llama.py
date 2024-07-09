from argparse import ArgumentParser
import torch
import llama
import os
from PIL import Image, ImageDraw
import cv2
import json
from tqdm import tqdm
import numpy as np
import torch.nn as nn
parser = ArgumentParser()
parser.add_argument('--llama_dir', type=str, help='llama directory')
parser.add_argument('--adapter_dir', type=str, help='adapter directory')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--action', type=str, help='llama directory')
conf = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else "cpu"
llama_dir = conf.llama_dir
# print(conf.adapter_dir, llama_dir, device)
model, preprocess = llama.load(conf.adapter_dir, llama_dir, device)
model.to(device)
model.eval()
if 'ori' in conf.adapter_dir:
    prompt = llama.format_prompt('Specify the contact point and orientation of pushing the object.') # though it is called pushing, but the prediction is the same as manipulating. It is just aboout the naming for prompt during training.
else:
    prompt = llama.format_prompt('Specify the contact point and gripper direction of manipulating the object.')
record_names = os.listdir(conf.data_dir)
for record_name in tqdm(record_names):
    out_dir  = os.path.join(conf.out_dir,record_name)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    record_dir = os.path.join(conf.data_dir,record_name)
    rgb_dir = os.path.join(record_dir,'original_rgb.png')
    if not os.path.exists(rgb_dir):
        continue
    start_pixel = 0
    size=336
    img_1 = Image.fromarray(np.array(Image.open(rgb_dir).convert('RGB'))[start_pixel:start_pixel+336,start_pixel:start_pixel+336,:])
    img = preprocess(img_1).unsqueeze(0).to(device) 
    with torch.no_grad():
        result = model.generate(img, [prompt])[0]
    # print(result)
    with open(os.path.join(out_dir, 'prediction.json'), 'w') as fout:
        json.dump(result, fout)
