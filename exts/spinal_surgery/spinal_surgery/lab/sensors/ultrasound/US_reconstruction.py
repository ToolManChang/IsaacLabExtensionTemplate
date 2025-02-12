from spinal_surgery.lab.sensors.US_slicer import USSlicer

class USReconstructor(USSlicer):

    def __init__(self, down_factor, us_cfg, label_maps, human_list, num_envs, x_z_range, init_x_z_x_angle, device, label_convert_map,
                 img_size, img_res, label_res=0.0015, max_distance=0.015, # [mm]
                 body_label=120, height = 0.12, height_img = 0.12,
                 visualize=True, plane_axes={'h': [0, 0, 1], 'w': [1, 0, 0]}):
        '''
        reconstruct 3D tensor from 2D images
        '''
        super().__init__(us_cfg, label_maps, human_list, num_envs, x_z_range, init_x_z_x_angle, device, label_convert_map,
                 img_size, img_res, label_res, max_distance,
                 body_label, height, height_img,
                 visualize, plane_axes)
        
        # define the position of world frame tensor (for now, we assume the world frame is at the origin)
        # store the all past image coordinates (downsampled) in human frame and their pixel values as a tensor
        
        
        