from spinal_surgery.lab.kinematics.human_frame_viewer import HumanFrameViewer
import torch
import numpy as np
from spinal_surgery.lab.kinematics.utils import *
from omni.isaac.lab.utils.math import combine_frame_transforms, transform_points, quat_from_matrix
import os
import time

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Vectorized version using torch.where for efficiency."""
    return torch.where(x > 0, torch.sqrt(x), torch.zeros_like(x))

@torch.jit.script
def quat_from_matrix_optimize(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions with optimized operations."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # Use torch.clamp for stability
    quat_candidates = quat_by_rijk / (2.0 * torch.clamp(q_abs[..., None], min=0.1)) # (B, 4, 4)
    argmax = q_abs.argmax(dim=-1)

    selected_quat = torch.gather(quat_candidates, dim=1, index=argmax[:, None, None].expand(-1, 1, 4)).squeeze(1)

    # Efficient one-hot indexing
    return selected_quat


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

            surface_normal_norms = torch.linalg.norm(surface_normal, dim=-1)
            surface_normal /= surface_normal_norms.unsqueeze(-1)
            self.surface_normal_list.append(surface_normal)


    def compute_world_ee_pose_from_cmd(self, world_to_human_pos, world_to_human_rot):
        '''
        compute world ee pose from human frame command
        world_to_human_pos: (num_envs, 3)
        world_to_human_rot: (num_envs, 4)
        '''

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
            # target_y_axis = target_y_axis / torch.linalg.norm(target_y_axis, dim=-1, keepdim=True)
            target_x_axis = torch.cross(target_y_axis, target_z_axis, dim=-1)
            target_rot_mat = torch.stack([target_x_axis, target_y_axis, target_z_axis], dim=-1) # (num_envs / n, 3, 3)
            target_quat = quat_from_matrix_optimize(target_rot_mat) # (num_envs / n, 4)
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
        
    