import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import json
import llama.utils
from llama import Tokenizer
import copy
import torchvision.transforms as transforms
import pandas as pd
import random
from random import randrange
import os
import numpy as np

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


# create data
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=(336, 336), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

class FinetuneDataset(Dataset):
    def __init__(self, config_path, args, max_words=30, tokenizer_path=None):
        print(f"read dataset config from {config_path}")
      
        self.mlm = args.mlm
        self.bins = args.bins
        self.config = config_path
        self.aff_prior = args.aff_prior
        
        ann = []
        for meta_name in os.listdir(self.config):
           
            meta_path = os.path.join(self.config, meta_name)
          
            ann.append(meta_path) 
            with open(meta_path, 'r') as f:
                meta_data = json.load(f)
          
        self.ann = ann
        print(f"total length: {len(self)}")
        
        self.transform = transform_train
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
       
       

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
       
        with open(self.ann[index], 'r') as f:
            data_item = json.load(f)
        filename = data_item['input']
        answer = data_item['conversations'][1]['gt']#value
        start_pixel = 0
        loc_tokens = []
        
        if self.bins == 'True' and  self.mlm == 'True' and self.aff_prior:
            words = answer.split(' ')
            for idx, word in enumerate(words):
                if '.' in word:
                    if '[' in word:
                        # print(word[1:-2])
                        words[idx] = '['+str(int(float(word[1:-2])//0.02)) + ','
                    elif ']' in word:
                        words[idx] = str(int(float(word[:-2])//0.02)) + ']'
                    else:
                        words[idx] = str(int(float(word[:-2])//0.02)) + ','
                    loc_tokens.append(idx)
                elif '(' in word:
                    loc_tokens.append(idx)
                    words[idx] = '('+str(int(word[1:-1])-start_pixel)+ ','
                elif ')' in word:
                    loc_tokens.append(idx)
                    words[idx] = str(int(word[:-2])-start_pixel)+ '),'
            answer = ' '.join([str(elem) for elem in words])

            i = random.randint(0, 3)
                
            #mlm and aff
            if i % 4 == 0:
                #finetune
                question = data_item['conversations'][0]['prompt']
                answer = answer
            elif i % 4 == 1:
                #mlm
                question_ori = answer.split(' ')
                i = random.sample(range(0, len(question_ori)-1), int(len(question_ori)*0.15))
                mask_loc = [loc_tokens[random.randint(0, len(loc_tokens)-1)],loc_tokens[random.randint(0, len(loc_tokens)-1)],loc_tokens[random.randint(0, len(loc_tokens)-1)]]
                question_mask = [word if idx not in mask_loc else "<mask>" for idx, word in enumerate(question_ori)]
                question = ' '.join([str(elem) for elem in question_mask])
                answer = answer
            elif i % 4 == 2:
                #affordance
                question = data_item['aff_question']
                answer = data_item['aff_gt']
            elif i % 4 == 3:
                #cat
                # question = data_item['conversations'][0]['prompt']
                # answer = answer
                question = data_item['cat_prompt']
                answer = data_item['cat_ans']

            image = Image.fromarray(np.array(Image.open(filename).convert('RGB'))[start_pixel:start_pixel+336,start_pixel:start_pixel+336,:])
            
            image = self.transform(image)
            format_instruction = question
            format_input = None
        
        input1 = llama.utils.format_prompt(format_instruction, format_input)
        input2 = input1 + answer
        
        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        
        return input2, labels, input2_mask, image
        
        
    


class PretrainDataset(Dataset):
    def __init__(self, config_path, transform, max_words=30, tokenizer_path=None):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        images, captions = [], []
        for meta_path in self.config['META']:
            images_this_meta, captions_this_meta = [], []
            for chunk in pd.read_csv(meta_path, sep='\t', lineterminator='\n', chunksize=10 ** 6):
                images_this_meta.extend(chunk['url'].tolist())
                captions_this_meta.extend(chunk['caption'].tolist())
            print(f"{meta_path}: len {len(images_this_meta)}")
            images.extend(images_this_meta)
            captions.extend(captions_this_meta)

        self.data_list = []
        for x, y in zip(images, captions):
            self.data_list.append({'url': x, 'caption': y})
        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        image_path, caption = sample['url'], sample['caption']
        if isinstance(caption, list):
            caption = random.choice(caption)
        caption = str(caption)

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        format_instruction = "Generate caption of this image"
        input1 = llama.utils.format_prompt(format_instruction, None)
        input2 = input1 + caption

        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, image