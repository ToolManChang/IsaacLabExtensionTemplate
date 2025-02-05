from spinal_surgery.lab.kinematics.surface_motion_planner import SurfaceMotionPlanner
import torch
import numpy as np
from spinal_surgery.lab.kinematics.utils import *
import cv2
import matplotlib.pyplot as plt
from omni.isaac.lab.utils.math import quat_from_matrix, combine_frame_transforms, transform_points, subtract_frame_transforms, matrix_from_quat

class LabelImgSlicer(SurfaceMotionPlanner):
    # Class: label image slicer
    # Function: __init__
    # Function: slice_label_img
    # Function: update_plotter
    def __init__(self, label_maps, human_list, num_envs, x_z_range, init_x_z_x_angle, device, label_convert_map,
                 img_size, img_res, label_res=0.0015, max_distance=0.015, # [m]
                 body_label=120, height = 0.1, height_img = 0.1,
                 visualize=True, plane_axes={'h': [0, 0, 1], 'w': [1, 0, 0]}):
        '''
        label maps: list of label maps (3D volumes)
        num_envs: number of environments
        plane_axes: dict of plane axes for imaging, in our case is 'x' and 'z' axes of the ee frame
        x_z_range: range of x and z in mm in the human frame as a rectangle [[min_x, min_z, min_x_angle], [max_x, max_z, max_x_angle]]
        init_x_z_y_angle: initial x, z human position, angle between ee x axis and human x axis in the xz plane 
        of human frame [x, z
        img_size: size of the image [w, h]
        img_res: resolution of the image
        ''' 
        super().__init__(label_maps, human_list, num_envs, x_z_range, init_x_z_x_angle, device, label_res, body_label, height, height_img, visualize, plane_axes)
        self.img_size = img_size
        self.img_res = img_res
        self.max_distance = max_distance
        self.img_real_size = [img_size[0] * img_res, img_size[1] * img_res]
        self.height_img = height_img
        for i in range(self.n_human_types):
            for key, value in label_convert_map.items():
                self.label_maps[i][self.label_maps[i] == key] = value

        # construct images
        self.label_img_tensor = torch.zeros((self.num_envs, self.img_size[0], self.img_size[1]), 
                                            dtype=torch.uint8, 
                                            device=self.device) # (num_envs, w, h)

        # construct grids
        self.x_grid, self.z_grid = torch.meshgrid(torch.arange(self.img_size[0], device=self.device) - self.img_size[0]//2, 
                                                  torch.arange(self.img_size[1], device=self.device))
        self.y_grid = torch.zeros_like(self.x_grid, device=self.device)
        self.img_coords = torch.stack([self.x_grid, self.y_grid, self.z_grid], dim=-1).reshape((-1, 3)).float() * img_res # (w * h, 3)

        return
    
    def get_human_img_coords(self, img_coords, world_to_human_pos, world_to_human_quat, world_to_ee_pos, world_to_ee_quat):
        human_to_ee_pos, human_to_ee_quat = subtract_frame_transforms(
            world_to_human_pos, world_to_human_quat, world_to_ee_pos, world_to_ee_quat) # (num_envs, 3), (num_envs, 4)
        human_to_ee_rot = matrix_from_quat(human_to_ee_quat) # (num_envs, 3, 3)
        normal_drcts = human_to_ee_rot[:, :, 2]
        prods = normal_drcts @ torch.tensor([0.0, 1.0, 0.0], device=self.device)
        normal_drcts[prods < 0, :] = -normal_drcts[prods < 0, :]
        human_to_img_pos = (human_to_ee_pos + self.height_img * normal_drcts) # (num_envs, 3)
        # human_to_img_pos = human_to_img_pos - self.img_real_size[0] / 2 * human_to_ee_rot[:, 0] # (num_envs, 3)
        human_img_coords = transform_points(img_coords, human_to_img_pos, human_to_ee_quat) # (num_envs, w*h, 3)
        human_img_coords = human_img_coords / self.label_res # convert to pixel coords
        # clamp the coords
        human_img_coords = torch.clamp(
            human_img_coords, 
            torch.zeros_like(human_img_coords, device=self.device), 
            max=torch.tensor(self.label_maps[0].shape, device=self.device).repeat(human_img_coords.shape[0], human_img_coords.shape[1], 1) - 1
        )
        return human_img_coords


    def slice_label_img(self, world_to_human_pos, world_to_human_quat, world_to_ee_pos, world_to_ee_quat):
        '''
        slice label image, x z correspond to w and h of the image
        world_to_human_pos: (num_envs, 3)
        world_to_human_rot: (num_envs, 4)
        world_to_ee_pos: (num_envs, 3)
        world_to_ee_quat: (num_envs, 4)
        '''
        self.human_img_coords = self.get_human_img_coords(
            self.img_coords,
            world_to_human_pos, 
            world_to_human_quat, 
            world_to_ee_pos, world_to_ee_quat) # (num_envs, w*h, 3)
        
        # speed up the slicing
        for i in range(self.n_human_types):
            self.label_img_tensor[i::self.n_human_types, :, :] = self.label_maps[i % self.n_human_types][
                self.human_img_coords[i::self.n_human_types, :, 0].int(), 
                self.human_img_coords[i::self.n_human_types, :, 1].int(), 
                self.human_img_coords[i::self.n_human_types, :, 2].int()
            ].reshape((-1, self.img_size[0], self.img_size[1]))

        # self.label_img_list = [
        #     self.label_maps[i % self.n_human_types][human_img_coords[i, :, 0].long(), human_img_coords[i, :, 1].long(), human_img_coords[i, :, 2].long()].reshape((self.img_size[0], self.img_size[1])) 
        #     for i in range(self.num_envs)]
        # smooth
        self.check_collision(self.label_img_tensor)
        
        return
    
    def get_distances_from_label_img(self, label_img_tensor):
        '''
        get distances from label image tensor (N, W, H)
        '''
        B, W, H = label_img_tensor.shape

        # Find the first nonzero index along the height dimension (axis=1)
        first_nonzero = torch.argmax((label_img_tensor > 0).int(), dim=2) # (N, W)
        return torch.min(first_nonzero, dim=1).values # (N, )

    def check_collision(self, label_img_tensor):
        '''
        check collision
        '''
        first_nonzero = self.get_distances_from_label_img(label_img_tensor)
        no_collide = first_nonzero > self.max_distance / self.label_res
        label_img_tensor[no_collide] = 0
    
    def visualize(self, first_n=20):
        '''
        visualize label image by combine'''
        first_n = min(first_n, self.num_envs)

        combined_img = self.label_img_tensor[:first_n, :, :].reshape((first_n * self.img_size[0], self.img_size[1])) # (w * first_n, h)

        combined_img_np = combined_img.cpu().numpy()

        cv2.imshow("Label Image Update", combined_img_np.T / np.max(combined_img_np))
        cv2.waitKey(1)

        return
