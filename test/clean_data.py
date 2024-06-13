import os
import shutil
data_dir = '/home/jiyao/mingxu/where2act-main/data/highreso996_1119'
data_tar = '/home/jiyao/mingxu/where2act-main/data/highreso996_1119_rgnonly'
data_list = os.listdir(data_dir)
for data_id in data_list:
    # file_list = os.listdir(os.path.join(data_dir,data_id))
    # for file_dir in file_list:
    #     if file_dir != 'rgb.png':
    #         os.remove(os.path.join(data_dir,data_id,file_dir))
    # if not os.path.exists(os.path.join(data_dir,data_id,'result.json')):
    #     shutil.rmtree(os.path.join(data_dir,data_id))
    source_file = os.path.join(data_dir,data_id,'rgb.png')
    destination_directory = os.path.join(data_tar,data_id)
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    shutil.copy(source_file, destination_directory)