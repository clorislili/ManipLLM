"""
    Environment with one object at center
        external: one robot, one camera
"""

from __future__ import division
import sapien.core as sapien
from sapien.core import Pose, SceneConfig, OptifuserConfig
from transforms3d.quaternions import axangle2quat, qmult
import numpy as np
from utils import process_angle_limit, get_random_number
from scipy.spatial.transform import Rotation as R
from PIL import Image
class ContactError(Exception):
    pass


class Env(object):
    
    def __init__(self, flog=None, show_gui=True, render_rate=20, timestep=1/500, \
            object_position_offset=0.0, succ_ratio=0.1):
        self.current_step = 0

        self.flog = flog
        self.show_gui = show_gui
        self.render_rate = render_rate
        self.timestep = timestep
        self.succ_ratio = succ_ratio
        self.object_position_offset = object_position_offset

        # engine and renderer
        self.engine = sapien.Engine(0, 0.001, 0.005)
        
        render_config = OptifuserConfig()
        render_config.shadow_map_size = 8192
        render_config.shadow_frustum_size = 10
        render_config.use_shadow = False
        render_config.use_ao = True
        
        self.renderer = sapien.OptifuserRenderer(config=render_config)
        self.renderer.enable_global_axes(False)
        
        self.engine.set_renderer(self.renderer)

        # GUI
        self.window = False
        if show_gui:
            self.renderer_controller = sapien.OptifuserController(self.renderer)
            self.renderer_controller.set_camera_position(-3.0+object_position_offset, 1.0, 3.0)
            self.renderer_controller.set_camera_rotation(-0.4, -0.8)

        # scene
        scene_config = SceneConfig()
        scene_config.gravity = [0, 0, -9.81]
        scene_config.solver_iterations = 20
        scene_config.enable_pcm = False
        scene_config.sleep_threshold = 0.0

        self.scene = self.engine.create_scene(config=scene_config)
        if show_gui:
            self.renderer_controller.set_current_scene(self.scene)

        self.scene.set_timestep(timestep)

        # add lights
        a,b,c = np.random.uniform(low=0.48, high=0.52, size=3)
        # self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.set_ambient_light([a, b, c])
        # d,e,f = np.random.uniform(low=0.45, high=0.55, size=3)
        self.scene.set_shadow_light([0, 1, -1], [0.5, 0.5, 0.5])
        self.scene.add_point_light([1+object_position_offset, 2, 2], [1, 1, 1])
        self.scene.add_point_light([1+object_position_offset, -2, 2], [1, 1, 1])
        self.scene.add_point_light([-1+object_position_offset, 0, 1], [1, 1, 1])

        # default Nones
        self.object = None
        self.object_target_joint = None

        # check contact
        self.check_contact = False

    def set_controller_camera_pose(self, x, y, z, yaw, pitch):
        self.renderer_controller.set_camera_position(x, y, z)
        self.renderer_controller.set_camera_rotation(yaw, pitch)
        self.renderer_controller.render()

    def load_object(self, scale, urdf, material, state='closed', category=None,primact_type=None):
        loader = self.scene.create_urdf_loader()
        loader.scale = scale
        # if 'Kettle' == category:
        #     loader.scale = 0.7
        # elif 'Stapler' == category:
        #     loader.scale = 0.7
        # if 'Kettle' == category:
        #     loader.scale = 0.7
        # elif 'Stapler' == category:
        #     loader.scale = 0.7
        
        self.object = loader.load(urdf, {"material": material})
        
        #self.object = loader.load(urdf, material)
        pose = Pose([self.object_position_offset, 0, 0], [1, 0, 0, 0])
        self.object.set_root_pose(pose)

        # compute link actor information
        self.all_link_ids = [l.get_id() for l in self.object.get_links()]
        self.movable_link_ids = []
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if category:
                    # print(j.type)
                    if 'Kettle' == category:
                        if 'pushing' == primact_type:
                            if str(j.type) == 'ArticulationJointType.PRISMATIC':
                                continue
                            else:
                                self.movable_link_ids.append(j.get_child_link().get_id())
                    if 'Toilet' == category:
                        if 'pushing' == primact_type:
                            if str(j.type) == 'ArticulationJointType.PRISMATIC':
                                continue
                            else:
                                self.movable_link_ids.append(j.get_child_link().get_id())
                    if 'wave' in category:
                        if 'pushing' == primact_type:
                            if str(j.type) == 'ArticulationJointType.REVOLUTE':
                                if (process_angle_limit(j.get_limits()[0, 0]) + process_angle_limit(j.get_limits()[0, 1])) == 0:
                                    continue
                                else:
                                    self.movable_link_ids.append(j.get_child_link().get_id())
                    if 'Wash' in category:
                        if 'pushing' == primact_type:
                            if str(j.type) == 'ArticulationJointType.REVOLUTE':
                                if (process_angle_limit(j.get_limits()[0, 0]) + process_angle_limit(j.get_limits()[0, 1])) == 0:
                                    continue
                                else:
                                    self.movable_link_ids.append(j.get_child_link().get_id())

                    else:
                        self.movable_link_ids.append(j.get_child_link().get_id())
                else:
                    self.movable_link_ids.append(j.get_child_link().get_id())
        
        if self.flog is not None:
            self.flog.write('All Actor Link IDs: %s\n' % str(self.all_link_ids))
            self.flog.write('All Movable Actor Link IDs: %s\n' % str(self.movable_link_ids))

        # set joint property
        for joint in self.object.get_joints():
            joint.set_drive_property(stiffness=0, damping=10)

        # set initial qpos
        joint_angles = []
        self.joint_angles_lower = []
        self.joint_angles_upper = []
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                # print(j.type)
                if category:
                    
                    if 'Kettle' == category:
                        if 'pushing' == primact_type:
                            if str(j.type) == 'ArticulationJointType.PRISMATIC':
                                continue
                
                l = process_angle_limit(j.get_limits()[0, 0])
                self.joint_angles_lower.append(float(l))
                r = process_angle_limit(j.get_limits()[0, 1])
                self.joint_angles_upper.append(float(r))
                # print(l,r)
                if state == 'closed':
                    # print('closed toilet')
                    joint_angles.append(float(l))
                elif state == 'open':
                    joint_angles.append(float(r))
                elif state == 'random-middle':
                    # print('middle')
                    if 'Kettle' == category:
                        if 'pushing' == primact_type:
                            joint_angles.append(float(0))
                    elif 'Laptop' == category:
                        if 'pushing' == primact_type:
                            # print('0.00000')
                            if r > 0.1:
                                joint_angles.append(float(0.2))
                            else:
                                joint_angles.append(float(0.0))
                    else:
                        joint_angles.append(float(get_random_number(l, r)))
                elif state == 'random-closed-middle':
                    if np.random.random() < 0.5:
                        joint_angles.append(float(get_random_number(l, r)))
                    else:
                        joint_angles.append(float(l))
                else:
                    raise ValueError('ERROR: object init state %s unknown!' % state)
        self.object.set_qpos(joint_angles)
        return joint_angles,l,r

    def set_object_joint_angles(self, joint_angles):
        self.object.set_qpos(joint_angles)

    def set_target_object_part_actor_id(self, actor_id):
        if self.flog is not None:
            self.flog.write('Set Target Object Part Actor ID: %d\n' % actor_id)
        self.target_object_part_actor_id = actor_id
        self.non_target_object_part_actor_id = list(set(self.all_link_ids) - set([actor_id]))
        self.actor_id = actor_id
        # get the link handler
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == actor_id:
                    self.target_object_part_actor_link = j.get_child_link()
        
        # moniter the target joint
        idx = 0
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == actor_id:
                    self.target_object_part_joint_id = idx
                idx += 1

    def get_object_qpos(self):
        return self.object.get_qpos()

    def get_target_part_qpos(self):
        qpos = self.object.get_qpos()
        for j in self.object.get_joints():
            if j.get_child_link().get_id() == self.actor_id:
                l = j.get_limits()[0, 0]
                r = j.get_limits()[0, 1]
                    
        return float(qpos[self.target_object_part_joint_id]),l,r
    
    def get_target_part_pose(self):
        
        return self.target_object_part_actor_link.get_pose()

    def start_checking_contact(self, robot_hand_actor_id, robot_gripper_actor_ids, strict):
        self.check_contact = True
        self.check_contact_strict = strict
        self.first_timestep_check_contact = True
        self.robot_hand_actor_id = robot_hand_actor_id
        self.robot_gripper_actor_ids = robot_gripper_actor_ids

    def get_material(self, static_friction, dynamic_friction, restitution):
        return self.engine.create_physical_material(static_friction, dynamic_friction, restitution)

    def render(self):
        if self.show_gui and (not self.window):
            self.window = True
            self.renderer_controller.show_window()
        self.scene.update_render()
        if self.show_gui and (self.current_step % self.render_rate == 0):
            self.renderer_controller.render()

    def step(self):
        contact = 0
        self.current_step += 1
        self.scene.step()
        if self.check_contact:
            if not self.check_contact_is_valid():
                contact += 1
                # print('contact!!!')
                # raise ContactError()
        return contact
    # check the first contact: only gripper links can touch the target object part link
    def check_contact_is_valid(self):
        self.contacts = self.scene.get_contacts()
        contact = False; valid = False; 
        for c in self.contacts:
            aid1 = c.actor1.get_id()
            aid2 = c.actor2.get_id()
            has_impulse = False
            for p in c.points:
                if abs(p.impulse @ p.impulse) > 1e-4:
                    has_impulse = True
                    break
            if has_impulse:
                if (aid1 in self.robot_gripper_actor_ids and aid2 == self.target_object_part_actor_id) or \
                   (aid2 in self.robot_gripper_actor_ids and aid1 == self.target_object_part_actor_id):
                       contact, valid = True, True
                if (aid1 in self.robot_gripper_actor_ids and aid2 in self.non_target_object_part_actor_id) or \
                   (aid2 in self.robot_gripper_actor_ids and aid1 in self.non_target_object_part_actor_id):
                    if self.check_contact_strict:
                        return False
                    else:
                        contact, valid = True, True
                if (aid1 == self.robot_hand_actor_id or aid2 == self.robot_hand_actor_id):
                    if self.check_contact_strict:
                        return False
                    else:
                        contact, valid = True, True
                # starting pose should have no collision at all
                if (aid1 in self.robot_gripper_actor_ids or aid1 == self.robot_hand_actor_id or \
                    aid2 in self.robot_gripper_actor_ids or aid2 == self.robot_hand_actor_id) and self.first_timestep_check_contact:
                        return False

        self.first_timestep_check_contact = False
        if contact and valid:
            self.check_contact = False
        return True

    def close_render(self):
        if self.window:
            self.renderer_controller.hide_window()
        self.window = False
    
    def wait_to_start(self):
        #assert(0)
        print('press q to start\n')
        while not self.renderer_controller.should_quit:
            self.scene.update_render()
            if self.show_gui:
                self.renderer_controller.render()

    def close(self):
        if self.show_gui:
            self.renderer_controller.set_current_scene(None)
        self.scene = None

    def load_robot(self,urdf_dir, scale):
        loader = self.scene.create_urdf_loader()
        loader.scale = scale
        self.robot = loader.load(urdf_dir)
        # self.robot.set_pose(Pose([-1.5, 0, 0]))
        self.robot.set_pose(Pose([-1.5, 0, 0]))
        self.robot.set_qpos(np.array([0, 0, -0.4, 0, 0.4, 0]))
        for joint in self.robot.get_active_joints():
            joint.set_drive_property(1e6, 1e4, 1e3)
        self.robot.set_drive_target(self.robot.get_qpos())
        return self.robot

    def quaternion_to_rotation_matrix(self, quaternion):
        r = R.from_quat(quaternion)
        rotation_matrix = r.as_matrix()
        return rotation_matrix

    def rotation_matrix_to_euler_angles(self, rotation_matrix):
        r = R.from_matrix(rotation_matrix)
        euler_angles = r.as_euler('xyz')  # 或者使用其他旋转顺序，如 'zyx'，取决于你的需求
        return euler_angles
    def compute_inverse_kinematics(self, delta_pose_world, ee_jacobian, damping=0.05):
        lmbda = np.eye(6) * (damping ** 2)
        delta_qpos = ee_jacobian.T @ \
                    np.linalg.lstsq(ee_jacobian.dot(ee_jacobian.T) + lmbda, delta_pose_world, rcond=None)[0]

        return delta_qpos

    def move_robot(self, env, robot: sapien.Articulation, start_pose: np.ndarray, ee_link: np.ndarray, num_steps):
        delta_pos = (start_pose.p - ee_link.get_pose().p ) / num_steps
        delta_rot = -(start_pose.q - ee_link.get_pose().q ) / num_steps
        target_rotation_matrix = self.quaternion_to_rotation_matrix(start_pose.q)
        current_rotation_matrix = self.quaternion_to_rotation_matrix(ee_link.get_pose().q)
        rotation_matrix_error = target_rotation_matrix @ np.linalg.inv(current_rotation_matrix)
        delta_rot = self.rotation_matrix_to_euler_angles(rotation_matrix_error) / num_steps
        delta_pos = (start_pose.p - ee_link.get_pose().p ) / num_steps
        for k in range(num_steps):
            delta_pose = np.concatenate([delta_pos, delta_rot])
            jacobian = self.robot.compute_world_cartesian_jacobian()[(len(self.robot.get_links()) - 1) * 6 - 6:(len(self.robot.get_links()) - 1) * 6]
            delta_qpos = self.compute_inverse_kinematics(delta_pose, jacobian)
            self.robot.set_drive_target(self.robot.get_drive_target() + delta_qpos)
            self.robot.set_qf(self.robot.compute_passive_force(external=False))
            env.step()
            # scene.up .date_render()
            env.render()
        return

    def move_robot_viz(self, camera, env, robot: sapien.Articulation, start_pose: np.ndarray, ee_link: np.ndarray, num_steps):
        delta_pos = (start_pose.p - ee_link.get_pose().p ) / num_steps
        delta_rot = -(start_pose.q - ee_link.get_pose().q ) / num_steps
        target_rotation_matrix = self.quaternion_to_rotation_matrix(start_pose.q)
        current_rotation_matrix = self.quaternion_to_rotation_matrix(ee_link.get_pose().q)
        rotation_matrix_error = target_rotation_matrix @ np.linalg.inv(current_rotation_matrix)
        delta_rot = self.rotation_matrix_to_euler_angles(rotation_matrix_error) / num_steps
        delta_pos = (start_pose.p - ee_link.get_pose().p ) / num_steps
        imgs = []
        for k in range(num_steps):
            delta_pose = np.concatenate([delta_pos, delta_rot])
            jacobian = self.robot.compute_world_cartesian_jacobian()[(len(self.robot.get_links()) - 1) * 6 - 6:(len(self.robot.get_links()) - 1) * 6]
            delta_qpos = self.compute_inverse_kinematics(delta_pose, jacobian)
            self.robot.set_drive_target(self.robot.get_drive_target() + delta_qpos)
            self.robot.set_qf(self.robot.compute_passive_force(external=False))
            env.step()
            # scene.up .date_render()
            env.render()
            rgb_pose, _ = camera.get_observation()
            fimg = (rgb_pose*255).astype(np.uint8)
            fimg = Image.fromarray(fimg)
            imgs.append(fimg)
        return imgs

    def create_drive(self, env, pose, ee_link):
        relative_pose = ee_link.get_pose().inv() * pose
        suction_drive = env.scene.create_drive(
                ee_link,
                relative_pose,
                env.target_object_part_actor_link,
                env.get_target_part_pose(),
            )
        suction_drive.set_x_properties(1e4, 1e2, 1e2)
        suction_drive.set_y_properties(1e4, 1e2, 1e2)
        suction_drive.set_z_properties(1e4, 1e2, 1e2)

