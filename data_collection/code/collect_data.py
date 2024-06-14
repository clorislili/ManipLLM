"""
    For panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
        50% all parts closed, 50% middle (for each part, 50% prob. closed, 50% prob. middle)
        Simulate until static before starting
"""

import os
import sys
import shutil
import numpy as np
from PIL import Image, ImageDraw
from utils import get_global_position_from_camera, save_h5
import cv2
import json
from argparse import ArgumentParser

from sapien.core import Pose
from env import Env, ContactError
from camera import Camera
from robots.panda_robot import Robot

parser = ArgumentParser()
parser.add_argument('shape_id', type=str)
parser.add_argument('category', type=str)
parser.add_argument('cnt_id', type=int)
parser.add_argument('primact_type', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--trial_id', type=int, default=0, help='trial id')
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
args = parser.parse_args()

def save_draw_point(img_arr, point, file_path):
    img = Image.fromarray((img_arr * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    radius = 2
    y,x=point
    draw.ellipse((y-radius,x-radius,y+radius,x+radius),fill='red')
    img.save(file_path)  

def save_draw_line(img_arr, p1, p2, file_path, fill='blue', width=2):
    img = Image.fromarray((img_arr * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.line([p1, p2], fill=fill, width=width)  
    y,x=p1
    radius=2
    draw.ellipse((y-radius,x-radius,y+radius,x+radius),fill='red')
    img.save(file_path)  

def generate_aff(object_link_ids,env,cam,cam_XYZA_world):
    aff_gt_all = []
    for i in range(len(object_link_ids)):
        action_type, hinge_pose, joint = env.set_target_object_part_actor_id(object_link_ids[i])
        pose = joint.get_parent_link().pose * joint.get_pose_in_parent_frame()
        axis_direct = pose.to_transformation_matrix()[:3,:3] @ [1,0,0]
        axis_direct /= np.linalg.norm(axis_direct)
        hinge_point = hinge_pose.p

        hinge_point_img = point_camera3d_to_img2d(point_world3d_to_camera3d(hinge_point, cam), cam)
        save_draw_point(rgb, hinge_point_img, os.path.join(out_dir, f'hinge_point_{i}.png'))

        q_point = hinge_point - axis_direct
        q_img = point_camera3d_to_img2d(point_world3d_to_camera3d(q_point, cam), cam)
        save_draw_line(rgb, hinge_point_img, q_img, os.path.join(out_dir, f'hinge_line_{i}.png'))

        part_movable_link_mask = cam.get_movable_link_mask([object_link_ids[i]])
       
        if str(action_type).split('.')[-1] == 'PRISMATIC':
            out_info['action_type_{}'.format(str(i))] = 'PRISMATIC'
            aff_gt = (part_movable_link_mask > 0).astype(np.uint8) * 255
            aff_gt_all.append(aff_gt)


        elif str(action_type).split('.')[-1] == 'REVOLUTE':
            out_info['action_type_{}'.format(str(i))] = 'REVOLUTE'
            indices_of_ones = np.where(part_movable_link_mask == 1) 
            sampled_indices = np.random.choice(np.arange(len(indices_of_ones[0])), size=len(indices_of_ones[0]), replace=False)  # 打乱这些像素的顺序
            sampled_points = np.vstack((indices_of_ones[0][sampled_indices], indices_of_ones[1][sampled_indices])).T  # 按照新循序把点排为 (X, 2) 的数组
            black_aff = np.zeros((336, 336))
            

            for index in range(len(sampled_indices)): 
                cur_index = sampled_points[index] 
                cur_point = cam_XYZA_world[cur_index[0],cur_index[1]] 
                point_rotated, flow_norm = rotate_point_around_axis(cur_point[:3], hinge_point, axis_direct)
                black_aff[cur_index[0],cur_index[1]] = flow_norm
               
                
            
            non_zero_values = black_aff[black_aff != 0]
            if non_zero_values.size > 0:
                min_value = np.min(non_zero_values)
                max_value = np.max(non_zero_values)
                normalized_flow = (black_aff - min_value) / (max_value - min_value)
            else:
                normalized_flow = np.zeros_like(black_aff)

            normalized_flow *= (part_movable_link_mask > 0)

            aff_gt = (normalized_flow * 255).astype(np.uint8)

            aff_gt_all.append(aff_gt)
    return aff_gt_all

def rotate_point_around_axis(point, axis_point, axis_direct):
    
    v1, v2, v3 = point
    c1, c2, c3 = axis_point
    d1, d2, d3 = axis_direct

    # calculate the rotation angle
    theta = np.arccos(d1) if d2 == 0 else np.arccos(d2) if d1 == 0 else np.arccos(d3)
    if theta == 0:
        return point, 0

    # calculate the rotation matrix
    R_mat = np.array([[np.cos(theta) + d1**2 * (1 - np.cos(theta)), d1 * d2 * (1 - np.cos(theta)) - d3 * np.sin(theta), d1 * d3 * (1 - np.cos(theta)) + d2 * np.sin(theta)],
                      [d2 * d1 * (1 - np.cos(theta)) + d3 * np.sin(theta), np.cos(theta) + d2**2 * (1 - np.cos(theta)), d2 * d3 * (1 - np.cos(theta)) - d1 * np.sin(theta)],
                      [d3 * d1 * (1 - np.cos(theta)) - d2 * np.sin(theta), d3 * d2 * (1 - np.cos(theta)) + d1 * np.sin(theta), np.cos(theta) + d3**2 * (1 - np.cos(theta))]])

    # calculate the translated vector
    T = np.array([c1, c2, c3])

    # calculate the rotated point
    P = np.array([v1, v2, v3])
    P_rot = np.dot(R_mat, P - T)
    P_rotated = P_rot + T

    # calculate the flow norm
    flow_norm = np.linalg.norm(P_rotated - P)

    return P_rotated, flow_norm

def point_camera3d_to_img2d(point_cam, cam):
    point_img = [-point_cam[1], -point_cam[2], point_cam[0]]
    point_img = np.dot((cam.get_metadata())["camera_matrix"][:3,:3], point_img)
    point_img = (point_img / point_img[2])[:2]
    point_img = (int(point_img[0]), int(point_img[1]))
    return point_img

def point_world3d_to_camera3d(point_world, cam):
    point_cam = np.linalg.inv(cam.get_metadata()['mat44']) @ np.append(point_world, 1)
    point_cam = point_cam[:3]
    return point_cam

shape_id = args.shape_id
trial_id = args.trial_id
primact_type = args.primact_type
if args.no_gui:
    out_dir = os.path.join(args.out_dir, '%s_%s_%d_%s_%d' % (shape_id, args.category, args.cnt_id, primact_type, trial_id))


if not os.path.exists(out_dir):
    os.makedirs(out_dir)
out_info = dict()

if args.random_seed is not None:
    np.random.seed(args.random_seed)
    out_info['random_seed'] = args.random_seed

# setup env
env = Env(show_gui=(not args.no_gui))

# setup camera
cam = Camera(env, image_size=336, random_position=True) #the visual encoder needs 336x336 input size
out_info['camera_metadata'] = cam.get_metadata_json()


# load shape
object_urdf_fn = '../asset/original_sapien_dataset/%s/mobility.urdf' % shape_id
object_material = env.get_material(4, 4, 0.01)
if primact_type == 'pulling':
    state = 'random-closed-middle'
    if np.random.random() < 0.8:
        state = 'closed'

#set object scale
out_info['object_state'] = state
scale = np.random.uniform(low=0.7, high=1.2)
out_info['scale'] = scale

#set object part angle
joint_angles = env.load_object(object_urdf_fn, object_material, scale,state=state)
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
    print("delete: object not still")
    shutil.rmtree(out_dir)
    env.close()
    exit(1)

#capture the original state of object
rgb, depth = cam.get_observation()
Image.fromarray((rgb*255).astype(np.uint8)).save(os.path.join(out_dir, 'original_rgb.png'))
img = Image.fromarray((rgb*255).astype(np.uint8))
draw = ImageDraw.Draw(img)

#generate the corresponding point cloud
cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
cam_XYZA_pts1 = np.ones((cam_XYZA_pts.shape[0],4))
cam_XYZA_pts1[:,:3] = cam_XYZA_pts 
cam_XYZA_pts_world = (cam.get_metadata()['mat44'] @ cam_XYZA_pts1.T).T
cam_XYZA_world = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts_world[:,:3], depth.shape[0], depth.shape[1])

#capture the surface norm
gt_nor = cam.get_normal_map()
Image.fromarray(((gt_nor+1)/2*255).astype(np.uint8)).save(os.path.join(out_dir, 'gt_nor.png'))

object_link_ids = env.movable_link_ids
gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)

Image.fromarray((gt_movable_link_mask>0).astype(np.uint8)*255).save(os.path.join(out_dir, 'interaction_mask.png'))
# try:
aff_gt_all = generate_aff(object_link_ids,env,cam,cam_XYZA_world)
# except:
    # shutil.rmtree(out_dir)
    # print('delete: affordance generation failed')
    # env.close()
    # exit()
if len(aff_gt_all) == 0:
    shutil.rmtree(out_dir)
    print('delete: affordance generation is 0')
    env.close()
    exit()
else:
    result_aff = aff_gt_all[0]
    for mask in aff_gt_all[1:]:
        result_aff = result_aff + mask
    Image.fromarray(result_aff).save(os.path.join(out_dir, 'aff_gt_all.png'))  
# randomly sample a pixel to interact
# xs, ys = np.where(gt_movable_link_mask>0)
xs,ys = np.where((result_aff/255)>0.6)
if len(xs) == 0:
    print("delete: not movable pixel on the object")
    shutil.rmtree(out_dir)
    env.close()
    exit(1)
idx = np.random.randint(len(xs))
x, y = xs[idx], ys[idx]
radius = 2
draw.ellipse((y-radius,x-radius,y+radius,x+radius),fill='red')
img.save(os.path.join(out_dir, 'contact_point.png'))
out_info['pixel_locs'] = [int(x), int(y)]
env.set_target_object_part_actor_id(object_link_ids[gt_movable_link_mask[x, y]-1])
out_info['target_object_part_actor_id'] = env.target_object_part_actor_id
out_info['target_object_part_joint_id'] = env.target_object_part_joint_id
part_movable_link_mask = cam.get_movable_link_mask([object_link_ids[gt_movable_link_mask[x, y]-1]])


norm_cam = gt_nor[x, y, :3]
norm_cam /= np.linalg.norm(norm_cam)
out_info['norm_cam'] = norm_cam.tolist()
norm_world = cam.get_metadata()['mat44'][:3, :3] @ norm_cam
out_info['norm_world'] = norm_world.tolist()


# the gripper z-axis direction is equal to the opposite of norm
action_direction_cam = -gt_nor[x, y, :3]
action_direction_cam /= np.linalg.norm(action_direction_cam)
out_info['gripper_direction_camera'] = action_direction_cam.tolist()
action_direction_world = cam.get_metadata()['mat44'][:3, :3] @ action_direction_cam
out_info['gripper_direction_world'] = action_direction_world.tolist()

# get pixel 3D position (cam2world)
position_cam = cam_XYZA[x, y, :3]
out_info['position_cam'] = position_cam.tolist()
position_cam_xyz1 = np.ones((4), dtype=np.float32)
position_cam_xyz1[:3] = position_cam
position_world_xyz1 = cam.get_metadata()['mat44'] @ position_cam_xyz1
position_world = position_world_xyz1[:3]
out_info['position_world'] = position_world.tolist()

# compute the rotnat
def add_noise(vector, noise_level=0.01):
    noise = np.random.normal(-noise_level, noise_level, vector.shape)
    return vector + noise

def orthogonalize_and_normalize(v1, v2):
    v1 /= np.linalg.norm(v1)
    v2 -= np.dot(v2, v1) * v1
    v2 /= np.linalg.norm(v2)
    return v1, v2

up = np.array(action_direction_world, dtype=np.float32)
up /= np.linalg.norm(up)

up = add_noise(up)
up /= np.linalg.norm(up)

forward = np.random.randn(3).astype(np.float32)
forward /= np.linalg.norm(forward)

up, forward = orthogonalize_and_normalize(up, forward)

left = np.cross(up, forward)
left /= np.linalg.norm(left)

forward = np.cross(left, up)
forward /= np.linalg.norm(forward)

left = np.cross(up, forward)
left /= np.linalg.norm(left)

out_info['gripper_forward_direction_world'] = forward.tolist()
forward_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ forward
out_info['gripper_forward_direction_camera'] = forward_cam.tolist()
out_info['gripper_up_direction_world'] = up.tolist()
up_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ up
out_info['gripper_up_direction_camera'] = up_cam.tolist()

#set start pose, contact pose, and pulling pose
rotmat = np.eye(4).astype(np.float32)
rotmat[:3, 0] = forward
rotmat[:3, 1] = left
rotmat[:3, 2] = up

contact_dist = 0.11 #the length of end effector finger

contact_rotmat = np.array(rotmat, dtype=np.float32)
contact_rotmat[:3, 3] = position_world - up * contact_dist
contact_pose = Pose().from_transformation_matrix(contact_rotmat)
out_info['contact_rotmat_world'] = contact_rotmat.tolist()

start_rotmat = np.array(rotmat, dtype=np.float32)
start_rotmat[:3, 3] = position_world - up * 0.2
start_pose = Pose().from_transformation_matrix(start_rotmat)
out_info['start_rotmat_world'] = start_rotmat.tolist()



pull_rotmat = np.array(rotmat, dtype=np.float32)
pull_rotmat[:3, 3] = position_world - up * 0.5
out_info['end_rotmat_world'] = pull_rotmat.tolist()


### load the end effector
robot_urdf_fn = './robots/panda_gripper.urdf'
robot_material = env.get_material(4, 4, 0.01)
robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=('pulling' in primact_type))

# move
robot.robot.set_root_pose(start_pose)
env.render()
rgb_final_pose, _ = cam.get_observation()
Image.fromarray((rgb_final_pose*255).astype(np.uint8)).save(os.path.join(out_dir, 'viz_start_pose.png'))

# env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, 'pushing' in primact_type)
out_info['start_target_part_qpos'] = env.get_target_part_qpos()
target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
position_local_xyz1 = np.linalg.inv(target_link_mat44) @ position_world_xyz1

# try:
robot.close_gripper()
# approach
robot.move_to_target_pose(contact_rotmat, 2000)
robot.wait_n_steps(2000)
rgb_final_pose, _ = cam.get_observation()
Image.fromarray((rgb_final_pose*255).astype(np.uint8)).save(os.path.join(out_dir, 'viz_contact_pose.png'))

#formulate suction
suction_drive = env.scene.create_drive(
            robot.robot.get_links()[-1],
            robot.robot.get_links()[-1].get_cmass_local_pose(),
            env.target_object_part_actor_link,
            env.target_object_part_actor_link.get_cmass_local_pose(),
        )
suction_drive.set_x_properties(stiffness=45000, damping=0)
suction_drive.set_y_properties(stiffness=45000, damping=0)
suction_drive.set_z_properties(stiffness=45000, damping=0)

#after stick the object, pull it
robot.move_to_target_pose(pull_rotmat, 2000)
robot.wait_n_steps(2000)
# except:
#     print('delete: contact error occur')
#     shutil.rmtree(out_dir)
#     env.close()
#     exit(1)


rgb_final_pose, final_depth = cam.get_observation()
Image.fromarray((rgb_final_pose*255).astype(np.uint8)).save(os.path.join(out_dir, 'viz_pull_pose.png'))
target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
position_world_xyz1_end = target_link_mat44 @ position_local_xyz1

out_info['touch_position_world_xyz_start'] = position_world_xyz1[:3].tolist()
out_info['touch_position_world_xyz_end'] = position_world_xyz1_end[:3].tolist()




out_info['result'] = 'VALID'
out_info['final_target_part_qpos'] = env.get_target_part_qpos()
#check if the manipulation is success
abs_motion = abs(out_info['final_target_part_qpos'] - out_info['start_target_part_qpos'])
j = out_info['target_object_part_joint_id']
tot_motion = out_info['joint_angles_upper'][j] - out_info['joint_angles_lower'][j] + 1e-8
mov_dir = np.array(out_info['touch_position_world_xyz_end'], dtype=np.float32) - \
                np.array(out_info['touch_position_world_xyz_start'], dtype=np.float32) + [1e-8,1e-8,1e-8]
mov_dir /= np.linalg.norm(mov_dir)
intended_dir = -np.array(out_info['gripper_direction_world'], dtype=np.float32)
mani_success = (intended_dir @ mov_dir > 0.5) and ((abs_motion > 0.1) or (abs_motion / tot_motion > 0.5))
out_info['mani_succ'] = str(mani_success)


if mani_success:
    with open(os.path.join(out_dir, 'result.json'), 'w') as fout:
        json.dump(out_info, fout)
else:
    shutil.rmtree(out_dir)
    print('delete: manipulation fails')
    env.close()
    exit(1)

print('------------------------ manipulation result is',out_dir, mani_success)
env.close()


