from spinal_surgery.lab.kinematics.human_frame_viewer import HumanFrameViewer
import torch
import numpy as np
from spinal_surgery.lab.kinematics.utils import *
from omni.isaac.lab.utils.math import quat_from_matrix, combine_frame_transforms, transform_points
import os


class SurfaceMotionPlanner(HumanFrameViewer):
    # Class: surface motion planner
    # Function: __init__
    # Function: plan_motion
    def __init__(self, label_maps, human_list, num_envs, x_z_range, init_x_z_x_angle, device, 
                 label_res=0.0015, body_label=120, height = 0.1, height_img = 0.1,
                 visualize=True, plane_axes={'h': [0, 0, 1], 'w': [1, 0, 0]}):
        '''
        label maps: list of label maps (3D volumes)
        num_envs: number of environments
        plane_axes: dict of plane axes for imaging, in our case is 'x' and 'z' axes of the ee frame
        x_z_range: range of x and z in mm in the human frame as a rectangle [[min_x, min_z, min_x_angle], [max_x, max_z, max_x_angle]]
        init_x_z_y_angle: initial x, z human position, angle between ee x axis and human x axis in the xz plane 
        of human frame [x, z, x_angle]
        height: height of ee above the surface
        '''
        super().__init__(label_maps, num_envs, device, label_res, height_img, visualize, plane_axes)

        self.x_z_range = x_z_range
        self.current_x_z_x_angle_cmd = torch.tensor(init_x_z_x_angle, device=device).repeat(num_envs, 1)
        self.device = device
        self.body_label = body_label
        self.height = height
        self.human_list = human_list

        # construct surface map list X * Z: surface point at 2d postion
        self.surface_map_list = []
        for i in range(self.n_human_types):
            lowest_y_array_path = human_list[i] + '/body_lowest_y_array.pt'
            if os.path.exists(lowest_y_array_path):
                surface_map = torch.load(lowest_y_array_path)
            else:
                surface_map = construct_lowest_y_array(self.label_maps[i], self.body_label)
                torch.save(surface_map, lowest_y_array_path)
            self.surface_map_list.append(surface_map)
        # construct surface normal list X * Z * 3: surface normal at 2d postion
        self.surface_normal_list = [] 
        for i in range(self.n_human_types):
            surface_normal_array_path = human_list[i] + '/body_surface_normal_array.pt'
            if os.path.exists(surface_normal_array_path):
                surface_normal = torch.load(surface_normal_array_path)
            else:
                surface_normal = construct_boundary_normals_array(self.label_maps[i], self.surface_map_list[i], self.body_label)
                torch.save(surface_normal, surface_normal_array_path)
            self.surface_normal_list.append(surface_normal)


    def compute_world_ee_pose_from_cmd(self, world_to_human_pos, world_to_human_rot):
        '''
        compute world ee pose from human frame command
        world_to_human_pos: (num_envs, 3)
        world_to_human_rot: (num_envs, 4)
        '''
        
        # target_quat_list = []
        # target_pos_list = []
        # # compute z axis
        # for i in range(self.num_envs):
        #     cur_x = self.current_x_z_x_angle_cmd[i, 0]
        #     cur_z = self.current_x_z_x_angle_cmd[i, 1]
        #     # human to ee rotation
        #     target_x_axis_proj = torch.tensor(
        #         [torch.cos(self.current_x_z_x_angle_cmd[i,2]), 
        #         0, 
        #         torch.sin(self.current_x_z_x_angle_cmd[i, 2])], device=self.device)
        #     target_z_axis = self.surface_normal_list[i % self.n_human_types][cur_x.int(), cur_z.int()]
        #     # target_z_axis = torch.tensor([0, 1.0, 0.0], dtype=torch.float32, device=self.device)
        #     target_y_axis = torch.cross(target_z_axis, target_x_axis_proj)
        #     target_y_axis = target_y_axis / torch.norm(target_y_axis)
        #     target_x_axis = torch.cross(target_y_axis, target_z_axis)
        #     target_x_axis = target_x_axis / torch.norm(target_x_axis)
        #     target_rot_mat = torch.stack([target_x_axis, target_y_axis, target_z_axis], dim=1)
        #     target_quat = quat_from_matrix(target_rot_mat)
        #     target_quat_list.append(target_quat)

        #     # get human to ee position
        #     target_y = self.surface_map_list[i % self.n_human_types][cur_x.int(), cur_z.int()]
        #     target_pos = torch.tensor([cur_x, target_y, cur_z], device=self.device) * self.label_res
        #     target_pos_list.append(target_pos - target_z_axis * self.height)
        
        # human_to_ee_target_quat = torch.stack(target_quat_list) # (num_envs, 4)
        # human_to_ee_target_pos = torch.stack(target_pos_list) # (num_envs, 3)

        # vectorize
        human_to_ee_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        human_to_ee_target_quat = torch.zeros((self.num_envs, 4), device=self.device)
        for i in range(self.n_human_types):
            cur_x = self.current_x_z_x_angle_cmd[i::self.n_human_types, 0] # (num_envs / n, 3)
            cur_z = self.current_x_z_x_angle_cmd[i::self.n_human_types, 1] # (num_envs / n, 3)
            target_x_axis_proj = torch.stack(
                [torch.cos(self.current_x_z_x_angle_cmd[i::self.n_human_types, 2]), 
                torch.zeros_like(cur_x), 
                torch.sin(self.current_x_z_x_angle_cmd[i::self.n_human_types, 2])], dim=-1) # (num_envs / n, 3)
            target_z_axis = self.surface_normal_list[i][cur_x.int(), cur_z.int()] # (num_envs / n, 3)
            target_y_axis = torch.cross(target_z_axis, target_x_axis_proj, dim=-1) # (num_envs / n, 3)
            target_y_axis = target_y_axis / torch.linalg.norm(target_y_axis, dim=-1, keepdim=True)
            target_x_axis = torch.cross(target_y_axis, target_z_axis, dim=-1)
            target_rot_mat = torch.stack([target_x_axis, target_y_axis, target_z_axis], dim=-1) # (num_envs / n, 3, 3)
            target_quat = quat_from_matrix(target_rot_mat) # (num_envs / n, 4)
            # rotation
            human_to_ee_target_quat[i::self.n_human_types] = target_quat
            # position
            target_y = self.surface_map_list[i][cur_x.int(), cur_z.int()] # (num_envs / n)
            target_pos = torch.stack([cur_x, target_y, cur_z], dim=-1) * self.label_res
            human_to_ee_target_pos[i::self.n_human_types] = target_pos - target_z_axis * self.height

        # get world to human position
        world_to_ee_target_pos, world_to_ee_target_rot = combine_frame_transforms(
            world_to_human_pos, world_to_human_rot, human_to_ee_target_pos, human_to_ee_target_quat
        )

        return world_to_ee_target_pos, world_to_ee_target_rot
    
    def update_cmd(self, d_x_z_x_angle):
        '''
        update command
        d_x_z_x_angle: (num_envs, 3)
        '''
        self.current_x_z_x_angle_cmd += d_x_z_x_angle
        self.current_x_z_x_angle_cmd = torch.clamp(self.current_x_z_x_angle_cmd, 
                                                   torch.tensor(self.x_z_range[0], device=self.device),
                                                   torch.tensor(self.x_z_range[1], device=self.device))
        
    