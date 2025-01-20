from spinal_surgery.lab.sensors.ultrasound.label_img_slicer import LabelImgSlicer
from spinal_surgery.lab.sensors.ultrasound.simulate_US_conv import USSimulatorConv
import cv2
import numpy as np


class USSlicer(LabelImgSlicer):
    # Class: US slicer
    # Function: __init__
    # Function: slice_US
    # Function: update_plotter
    def __init__(self, us_cfg, label_maps, human_list, num_envs, x_z_range, init_x_z_x_angle, device, label_convert_map,
                 img_size, img_res, label_res=0.0015,
                 body_label=120, height = 0.1, height_img = 0.101,
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
        super().__init__(label_maps, human_list, num_envs, x_z_range, init_x_z_x_angle, device, label_convert_map,
                 img_size, img_res, label_res,
                 body_label, height, height_img,
                 visualize, plane_axes)
        self.us_sim = USSimulatorConv(us_cfg)


    def slice_US(self, world_to_human_pos, world_to_human_quat, world_to_ee_pos, world_to_ee_quat):
        self.slice_label_img(world_to_human_pos, world_to_human_quat, world_to_ee_pos, world_to_ee_quat)
        self.us_img_tensor = self.us_sim.simulate_US_image(self.label_img_tensor.permute(0, 2, 1), False) # (n, H, W)

    def visualize(self, first_n=20):
        super().visualize(first_n)
        first_n = min(first_n, self.num_envs)

        combined_US_img = self.us_img_tensor[:first_n, :, :].permute(0, 2, 1).reshape((first_n * self.img_size[0], self.img_size[1])) # (w * first_n, h)

        combined_img_np = combined_US_img.cpu().numpy()

        cv2.imshow("US Image Update", combined_img_np.T / 20)
        cv2.waitKey(1)


