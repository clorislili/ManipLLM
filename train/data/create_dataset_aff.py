import json
import os
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import argparse
from tqdm import tqdm

print('Start generating training json..............')
count = 0
parser = argparse.ArgumentParser()
parser.add_argument('--folder_dir', type=str, help='dataset dir')
parser.add_argument('--output_dir', type=str, help='training json dir')
parser.add_argument('--num_point', type=int, help='training json dir')
args = parser.parse_args()

folder_dir = args.folder_dir
folder_names = os.listdir(folder_dir)
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    print('json files already exists, beginning training')
    exit()
cal_cat = dict()

for item in tqdm(folder_names):
    NUM_OF_POINTS = args.num_point
    cur_dir = os.path.join(folder_dir,str(item))
    cat = item.split('_')[1]
    if os.path.exists(os.path.join(cur_dir, 'result.json')):
        with open(os.path.join(cur_dir, 'result.json'), 'r') as fin:
            data_inf = json.load(fin)
            if data_inf['mani_succ'] != 'True':
                continue
            
            aff_gt_dir = os.path.join(cur_dir, 'aff_gt_all.png')
            if not os.path.exists(aff_gt_dir):
                continue
            img_pil = Image.open(os.path.join(cur_dir, 'original_rgb.png'))
            intermask_pil = np.array(Image.open(os.path.join(cur_dir, 'interaction_mask.png')))
            gray_image = ImageOps.grayscale(img_pil)
            threshold = 200  # Adjust the threshold value as needed
            object_mask = gray_image.point(lambda p: p < threshold and 255)
            object_mask.save(os.path.join(cur_dir, 'object_mask.png'))
            
            object_mask = np.array(object_mask)/255
            
            
            aff_gt_pil = Image.open(aff_gt_dir)
            aff_gt = np.array(aff_gt_pil)/255
            result_mask = np.where(aff_gt < 0.2, intermask_pil, 0).astype(np.uint8)
            object_mask = np.where(aff_gt < 0.2, object_mask, 0).astype(np.uint8)
            Image.fromarray((result_mask).astype(np.uint8)).save(os.path.join(cur_dir, 'result_mask.png'))
            Image.fromarray((object_mask*255).astype(np.uint8)).save(os.path.join(cur_dir, 'object_mask.png'))
        
            row_indices_pos, col_indices_pos = np.where(aff_gt > 0.8)
            if NUM_OF_POINTS > len(row_indices_pos):
                NUM_OF_POINTS = len(row_indices_pos)
            
            row_indices_neg1, col_indices_neg1 = np.where(result_mask > 0.8)
            
           
            if NUM_OF_POINTS > len(row_indices_neg1) and len(row_indices_neg1) != 0:
                NUM_OF_POINTS = len(row_indices_neg1)
            
            if NUM_OF_POINTS == 0:
                continue
            
            if len(row_indices_neg1) != 0 :
                indices_neg = np.random.choice(len(row_indices_neg1), size=NUM_OF_POINTS//2, replace=False)
                selected_row_indices_neg = row_indices_neg1[indices_neg].reshape(-1, 1)
                selected_col_indices_neg = col_indices_neg1[indices_neg].reshape(-1, 1)
                top_indices_neg1 = np.hstack((selected_row_indices_neg, selected_col_indices_neg))
                top_indices_neg1_gt = np.zeros(top_indices_neg1.shape[0])
            
            row_indices_neg, col_indices_neg = np.where(object_mask > 0.8)
            
            if len(row_indices_neg) != 0 and len(row_indices_neg1) != 0:
                indices_neg = np.random.choice(len(row_indices_neg), size=NUM_OF_POINTS//2, replace=False)
                selected_row_indices_neg = row_indices_neg[indices_neg].reshape(-1, 1)
                selected_col_indices_neg = col_indices_neg[indices_neg].reshape(-1, 1)
                top_indices_neg2 = np.hstack((selected_row_indices_neg, selected_col_indices_neg))
                top_indices_neg2_gt = np.zeros(top_indices_neg2.shape[0])
            else:
                try:
                    indices_neg = np.random.choice(len(row_indices_neg), size=NUM_OF_POINTS, replace=False)
                    selected_row_indices_neg = row_indices_neg[indices_neg].reshape(-1, 1)
                    selected_col_indices_neg = col_indices_neg[indices_neg].reshape(-1, 1)
                    top_indices_neg2 = np.hstack((selected_row_indices_neg, selected_col_indices_neg))
                    top_indices_neg2_gt = np.zeros(top_indices_neg2.shape[0])
                except:
                    continue
            
            
            indices_pos = np.random.choice(len(row_indices_pos), size=NUM_OF_POINTS, replace=False)
            selected_row_indices_pos = row_indices_pos[indices_pos].reshape(-1, 1)
            selected_col_indices_pos = col_indices_pos[indices_pos].reshape(-1, 1)
            top_indices_pos = np.hstack((selected_row_indices_pos, selected_col_indices_pos))
            top_indices_pos_gt = np.ones(top_indices_pos.shape[0])

            if len(row_indices_neg1) == 0 :
                
                select_indices = np.vstack((top_indices_neg2,  top_indices_pos))
                select_indices_gt = np.concatenate((top_indices_neg2_gt, top_indices_pos_gt))
                
            else:
                
                select_indices = np.vstack((top_indices_neg1, top_indices_neg2,  top_indices_pos))
                select_indices_gt = np.concatenate((top_indices_neg1_gt, top_indices_neg2_gt, top_indices_pos_gt))
            
            permutation = np.random.permutation(len(select_indices_gt))
            select_indices = select_indices[permutation]
            select_indices_gt = select_indices_gt[permutation]
            
            mapping = {0: "no", 1: "yes"}
            if len(select_indices_gt) == 0:
                continue
            select_string_gt = np.vectorize(mapping.get)(select_indices_gt)
            
            
            select_string = np.array2string(select_indices, separator=',', formatter={'all': lambda x: str(x)})[1:-1].strip().replace("\n", " ")
            select_string_gt = np.array2string(select_string_gt, separator=',', formatter={'all': lambda x: str(x)})[1:-1].strip().replace("\n", " ")
            
            aff_question = 'Determine if operating on each following point can effectively manipulate the object within the image: {}'.format(select_string)
            aff_gt = select_string_gt
            
            
            #draw the selected point in the image
            draw = ImageDraw.Draw(img_pil)
            if len(row_indices_neg1) != 0 :
                for index in range(top_indices_neg1.shape[0]):
                    draw.point((top_indices_neg1[index][1],top_indices_neg1[index][0]),'blue')
            for index in range(top_indices_neg2.shape[0]):
                draw.point((top_indices_neg2[index][1],top_indices_neg2[index][0]),'blue')
            for index in range(top_indices_pos.shape[0]):
                draw.point((top_indices_pos[index][1],top_indices_pos[index][0]),'red')
            img_pil.save(os.path.join(cur_dir, 'select_point.png'))

            up_cam = data_inf['gripper_up_direction_camera']
            forward_cam = data_inf['gripper_forward_direction_camera']
            x,y = data_inf['pixel_locs']
            data = {
                
                "conversations": [
                    {
                        "prompt": "Specify the contact point and gripper direction of manipulating the object."
                    },
                    {
                        "gt": f"The contact point is ({int(x)}, {int(y)}),  the gripper up direction is {up_cam}, the gripper forward direction is {forward_cam}."

                    }
                ],
                'cat_prompt': 'What is the category of the object in the image?',
                'cat_ans': item.split('_')[1],
                "instruction": "Specify the contact point and gripper direction of manipulating the object.",
                "input": os.path.join(cur_dir, 'original_rgb.png'),
                'aff_question': aff_question,
                'aff_gt': aff_gt.strip()
                
            }
            if not os.path.exists(os.path.join(cur_dir, 'original_rgb.png')):
                continue
            
            json_data = json.dumps(data, indent=4)
            cat = item.split('_')[1]
            
            if cat not in list(cal_cat.keys()):
                cal_cat[cat] = 1
            else:
                if cal_cat[cat] > 900:
                    continue
                else:
                    cal_cat[cat] += 1
            
            
            with open(os.path.join(output_dir,'{}.json'.format(item)), "w") as file:
                file.write(json_data)
            
print('Numbers of each training category: ', cal_cat)
print('Finish generating training json..............')