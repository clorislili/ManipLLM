
import os
import sys
import shutil
from argparse import ArgumentParser
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn.functional as F
from sapien.core import Pose
from env_ori import Env,ContactError
from camera import Camera
from robots.panda_robot import Robot
import imageio
import cv2
import json
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import llama


parser = ArgumentParser()
parser.add_argument('--llama_dir', type=str, help='llama directory')
parser.add_argument('--adapter_dir', type=str,default='./', help='adapter directory')
parser.add_argument('--result_suffix', type=str, default='nothing')
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if out_dir exists [default: False]')

parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--record_name', type=str)
parser.add_argument('--out_dir', type=str)
eval_conf = parser.parse_args()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)




#previous info are saved in result.json
shape_id, category, cnt_id, primact_type, trial_id = eval_conf.record_name.split('_')

out_dir = os.path.join(eval_conf.out_dir, '%s_%s_%s_%s_%d' % (shape_id, category, cnt_id, primact_type, int(trial_id)))


flog = open(os.path.join(out_dir, 'log.txt'), 'w')
out_info = dict()
try:
    with open(os.path.join(eval_conf.data_dir, eval_conf.record_name, 'result.json'), 'r') as fin:
        replay_data = json.load(fin)
except:
    print('no replay data')
    exit(1)


env = Env(flog=flog, show_gui=(not eval_conf.no_gui))

# setup camera
cam_theta = replay_data['camera_metadata']['theta']
cam_phi = replay_data['camera_metadata']['phi']
cam_dist = replay_data['camera_metadata']['dist']
cam = Camera(env, theta=cam_theta, phi=cam_phi, dist=cam_dist)
out_info['camera_metadata_init'] = cam.get_metadata_json()


if not eval_conf.no_gui:
    env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi+cam_theta, -cam_phi)



# load shape
object_urdf_fn = '/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/xcy/where2act/data/where2act_original_sapien_dataset/%s/mobility.urdf' % shape_id
flog.write('object_urdf_fn: %s\n' % object_urdf_fn)
object_material = env.get_material(4, 4, 0.01)
state = replay_data['object_state']
flog.write('Object State: %s\n' % state)
out_info['object_state'] = state
scale = replay_data['scale']
env.load_object(scale, object_urdf_fn, object_material, state=state)
joint_angles = replay_data['joint_angles']
env.set_object_joint_angles(joint_angles)
out_info['joint_angles'] = joint_angles
out_info['joint_angles_lower'] = env.joint_angles_lower
out_info['joint_angles_upper'] = env.joint_angles_upper
cur_qpos = env.get_object_qpos()

# simulate some steps for the object to stay rest
still_timesteps = 0
wait_timesteps = 0
while still_timesteps < 5000 and wait_timesteps < 20000:
    env.step()
    env.render()
    cur_new_qpos = env.get_object_qpos()
    invalid_contact = False
    for c in env.scene.get_contacts():
        for p in c.points:
            if abs(p.impulse @ p.impulse) > 1e-4:
                invalid_contact = True
                break
        if invalid_contact:
            break
    if np.max(np.abs(cur_new_qpos - cur_qpos)) < 1e-6 and (not invalid_contact):
        still_timesteps += 1
    else:
        still_timesteps = 0
    cur_qpos = cur_new_qpos
    wait_timesteps += 1

if still_timesteps < 5000:
    printout(flog, 'Object Not Still!')
    flog.close()
    env.close()
    exit(1)

rgb, depth = cam.get_observation()
Image.fromarray((rgb*255).astype(np.uint8)).save(os.path.join(out_dir, 'rgb_img.png'))
img = Image.fromarray((rgb*255).astype(np.uint8))

gt_nor = cam.get_normal_map()
Image.fromarray(((gt_nor+1)/2*255).astype(np.uint8)).save(os.path.join(out_dir, 'gt_nor.png'))

object_link_ids = env.movable_link_ids
gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)
mask = (gt_movable_link_mask > 0)


if os.path.exists(os.path.join(out_dir, 'prediction.json')):
    with open(os.path.join(out_dir, 'prediction.json'), 'r') as fin:
        result = json.load(fin)
else:
    print('!!!!!!!!!!!!!!!!!!!!!!no prediction !!!!!!!!!!!!!!!!!!!!!!!!')
    flog.close()
    env.close()
    exit(2)


print('answer from model: ', result)

object_link_ids = env.movable_link_ids
gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)
x, y = result.split('(')[1].split(')')[0].split(', ')
x = int(x)
y = int(y)
if gt_movable_link_mask[x,y] == 0:
    exit()

norm_dir = gt_nor[x,y]

gt_nor = cam.get_normal_map()
Image.fromarray(((gt_nor+1)/2*255).astype(np.uint8)).save(os.path.join(out_dir, 'gt_nor.png'))

d_x, d_y, d_z = result.split('[')[1].split(']')[0].split(', ')
gripper_direction_camera = np.array([int(d_x)*0.02, int(d_y)*0.02, int(d_z)*0.02])
fd_x, fd_y, fd_z = result.split('[')[2].split(']')[0].split(', ')
gripper_forward_direction_camera = np.array([int(fd_x)*0.02, int(fd_y)*0.02, int(fd_z)*0.02])

draw = ImageDraw.Draw(img)
draw.point((y,x),'red')
img.save(os.path.join(out_dir, 'contact_point.png'))

cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts,out = cam.compute_camera_XYZA(depth)
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
position_cam = cam_XYZA[x, y, :3]

position_cam_xyz1 = np.ones((4), dtype=np.float32)
position_cam_xyz1[:3] = position_cam
position_world_xyz1 = cam.get_metadata()['mat44'] @ position_cam_xyz1
position_world = position_world_xyz1[:3]
target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
env.set_target_object_part_actor_id(target_part_id)
out_info['target_object_part_actor_id'] = env.target_object_part_actor_id
out_info['target_object_part_joint_id'] = env.target_object_part_joint_id


def plot_mani(cam,up, forward):
    # we use the norm of the contact point to correct tge z-axis of end-effector
    if (up @ norm_dir[:3] ) > 0:
        up = -up

    up /= np.linalg.norm(up)
    up = cam.get_metadata()['mat44'][:3,:3] @ up
    forward = cam.get_metadata()['mat44'][:3,:3] @ forward
    out_info['gripper_direction_world'] = up.tolist()
    
    up = np.array(up, dtype=np.float32)
    up /= np.linalg.norm(up)
    forward = np.array(forward, dtype=np.float32)
    forward /= np.linalg.norm(forward)
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    forward = np.cross(left, up)
    forward /= np.linalg.norm(forward)

    rotmat = np.eye(4).astype(np.float32)
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up
    
    final_rotmat = np.array(rotmat, dtype=np.float32)
    final_rotmat[:3, 3] = position_world - up * 0.1
    final_pose = Pose().from_transformation_matrix(final_rotmat)

    start_rotmat = np.array(rotmat, dtype=np.float32)
    start_rotmat[:3, 3] = position_world - up * 0.15
    # start_rotmat[:3, 3] = position_world
    start_pose = Pose().from_transformation_matrix(start_rotmat)

    pull_rotmat = np.array(rotmat, dtype=np.float32)
    pull_rotmat[:3, 3] = position_world - up * 0.5
    pull_pose = Pose().from_transformation_matrix(pull_rotmat)
    out_info['pull_rotmat_world'] = pull_rotmat.tolist()
    
    #load robot
    robot_urdf_fn = './robots/panda_gripper.urdf'
    robot_material = env.get_material(4, 4, 0.01)
    robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=('pulling' in primact_type))
    
    
    robot.robot.set_root_pose(start_pose)

    
    env.render()
    rgb_final_pose, _ = cam.get_observation()
    Image.fromarray((rgb_final_pose*255).astype(np.uint8)).save(os.path.join(out_dir, 'viz_start_pose.png'))
    
    
    
    out_info['start_target_part_qpos'],_,_ = env.get_target_part_qpos()
    
    
    success = True
    target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
    position_local_xyz1 = np.linalg.inv(target_link_mat44) @ position_world_xyz1
    
    
    robot.close_gripper()
    robot.wait_n_steps(2000)
    

    # approach
    robot.move_to_target_pose(final_rotmat, 2000)
    robot.wait_n_steps(2000)
    rgb_final_pose, _ = cam.get_observation()
    Image.fromarray((rgb_final_pose*255).astype(np.uint8)).save(os.path.join(out_dir, 'viz_mid_pose.png'))
    
    suction_drive = env.scene.create_drive(
                robot.robot.get_links()[-1],
                robot.robot.get_links()[-1].get_cmass_local_pose(),
                env.target_object_part_actor_link,
                env.target_object_part_actor_link.get_cmass_local_pose(),
            )
    suction_drive.set_x_properties(stiffness=45000, damping=0)
    suction_drive.set_y_properties(stiffness=45000, damping=0)
    suction_drive.set_z_properties(stiffness=45000, damping=0)

    
    if primact_type == 'pulling':
        robot.move_to_target_pose(pull_rotmat, 2000)
        robot.wait_n_steps(2000)


    
    target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
    position_world_xyz1_end = target_link_mat44 @ position_local_xyz1
    out_info['touch_position_world_xyz_start'] = position_world_xyz1[:3].tolist()
    out_info['touch_position_world_xyz_end'] = position_world_xyz1_end[:3].tolist()
    if success==True:
        succ=True
        out_info['final_target_part_qpos'],_,_ = env.get_target_part_qpos()
        print(out_info['final_target_part_qpos'],out_info['start_target_part_qpos'])
        abs_motion = abs(out_info['final_target_part_qpos'] - out_info['start_target_part_qpos'])
        j = out_info['target_object_part_joint_id']
        tot_motion = out_info['joint_angles_upper'][j] - out_info['joint_angles_lower'][j] + 1e-8
        mani_success = (abs_motion > 0.03) or (abs_motion / tot_motion > 0.5)
    else:
        mani_success = False
    if mani_success:
        if primact_type == 'pushing':
            mani_success = mani_success
        elif primact_type == 'pulling':
            mov_dir = np.array(out_info['touch_position_world_xyz_end'], dtype=np.float32) - \
                        np.array(out_info['touch_position_world_xyz_start'], dtype=np.float32)
            mov_dir /= np.linalg.norm(mov_dir)
            intended_dir = -np.array(out_info['gripper_direction_world'], dtype=np.float32)
            mani_success = (intended_dir @ mov_dir > 0.3)
    return success, mani_success

success, mani_succ = plot_mani(cam,gripper_direction_camera, gripper_forward_direction_camera)
out_info['succ'] = np.array(success, dtype=bool).tolist()
 
out_info['mani_succ'] = np.array(mani_succ, dtype=bool).tolist()
rgb_final_pose, _ = cam.get_observation()
Image.fromarray((rgb_final_pose*255).astype(np.uint8)).save(os.path.join(out_dir, 'viz_target_pose.png'))

print(success, mani_succ)
with open(os.path.join(out_dir, 'result2.json'), 'w') as fout:
    json.dump(out_info, fout)
    print(out_dir)
flog.close()
env.close()
