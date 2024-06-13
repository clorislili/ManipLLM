import os
import shutil
file_path = '../stats/train_30cats_train_data_list.txt'

# Open the file in read mode
lines = []
with open(file_path, 'r') as file:
    # Iterate over each line in the file
    for line in file:
        # Process the line (for example, print it)
        # print(line.strip())  # .strip() removes leading/trailing whitespace, including the newline character
        lines.append(line.strip())
data_dir = '../data/train_data'
target_dir = '../data/train_data0606'
data_list = os.listdir(data_dir)
cat_cal = dict()
for data_name in data_list:
    data_id = data_name.split('_')[0]
    data_cat = data_name.split('_')[1]
    source_dir = os.path.join(data_dir,data_name)
    destination_dir = os.path.join(target_dir,data_name)
    try:
        for line in lines:
            if data_id in line and data_cat in line:
                shutil.copytree(source_dir, destination_dir)
                break
    except:
        continue
    