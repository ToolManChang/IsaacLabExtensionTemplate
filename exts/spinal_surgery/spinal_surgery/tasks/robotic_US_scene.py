# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.managers import SceneEntityCfg
import nibabel as nib
import cProfile
import time
import numpy as np

##
# Pre-defined configs
##
from spinal_surgery.assets.kuka_US import *
from omni.isaac.lab.utils.math import subtract_frame_transforms, combine_frame_transforms
from pxr import Gf, UsdGeom
from scipy.spatial.transform import Rotation as R
from spinal_surgery.lab.kinematics.human_frame_viewer import HumanFrameViewer
from spinal_surgery.lab.kinematics.surface_motion_planner import SurfaceMotionPlanner
from spinal_surgery.lab.sensors.ultrasound.label_img_slicer import LabelImgSlicer
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer
from ruamel.yaml import YAML
from spinal_surgery import PACKAGE_DIR

scene_cfg = YAML().load(open(f"{PACKAGE_DIR}/tasks/cfgs/robotic_US_scene.yaml", 'r'))

# robot
robot_cfg = scene_cfg['robot']
INIT_STATE_ROBOT_US = ArticulationCfg.InitialStateCfg(
    joint_pos={
        "lbr_joint_0": robot_cfg['joint_pos'][0],
        "lbr_joint_1": robot_cfg['joint_pos'][1],
        "lbr_joint_2": robot_cfg['joint_pos'][2],
        "lbr_joint_3": robot_cfg['joint_pos'][3], # -1.2,
        "lbr_joint_4": robot_cfg['joint_pos'][4],
        "lbr_joint_5": robot_cfg['joint_pos'][5], # 1.5,
        "lbr_joint_6": robot_cfg['joint_pos'][6],
    },
    pos = robot_cfg['pos'] # ((0.0, -0.75, 0.4))
)

# patient
patient_cfg = scene_cfg['patient']
quat = R.from_euler("yxz", patient_cfg['euler_yxz'], degrees=True).as_quat()
INIT_STATE_HUMAN = RigidObjectCfg.InitialStateCfg(
    pos=patient_cfg['pos'], # 0.7
    rot=((quat[3], quat[0], quat[1], quat[2]))
)

# bed
bed_cfg = scene_cfg['bed']
quat = R.from_euler("xyz", bed_cfg['euler_xyz'], degrees=True).as_quat()
INIT_STATE_BED = AssetBaseCfg.InitialStateCfg(
    pos=bed_cfg['pos'], 
    rot=((quat[3], quat[0], quat[1], quat[2]))
)
scale_bed = bed_cfg['scale']
# use stl: Totalsegmentator_dataset_v2_subset_body_contact
human_usd_list = [
            f"{ASSETS_DATA_DIR}/HumanModels/Totalsegmentator_dataset_v2_subset_body_from_urdf/" + p_id for p_id in patient_cfg['id_list']
]
human_stl_list = [
            f"{ASSETS_DATA_DIR}/HumanModels/Totalsegmentator_dataset_v2_subset_stl/" + p_id for p_id in patient_cfg['id_list']
]
human_raw_list = [
            f"{ASSETS_DATA_DIR}/HumanModels/Totalsegmentator_dataset_v2_subset/" + p_id for p_id in patient_cfg['id_list']
]

usd_file_list = [human_file + "/combined_wrapwrap/combined_wrapwrap.usd" for human_file in human_usd_list]
label_map_file_list = [human_file + "/combined_label_map.nii.gz" for human_file in human_stl_list]
ct_map_file_list = [human_file + "/ct.nii.gz" for human_file in human_raw_list]

label_res = patient_cfg['label_res']
scale = 1/label_res

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    # kuka US
    robot_US = KUKA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_US",
        init_state=INIT_STATE_ROBOT_US
    )

    # medical bad
    medical_bad = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Bed", 
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSETS_DATA_DIR}/MedicalBed/usd_no_contact/hospital_bed.usd",
            scale = (scale_bed, scale_bed, scale_bed),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ), # Improves a lot of time count=8 0.014-0.013
        ),
        init_state = INIT_STATE_BED
    )


    # human: 
    human = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Human", 
        spawn=sim_utils.MultiUsdFileCfg(
        usd_path=usd_file_list,
        random_choice=False,
        scale = (label_res, label_res, label_res),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
           articulation_enabled=False,
           solver_position_iteration_count=8,
           solver_velocity_iteration_count=0,
        ),
        ),
        init_state = INIT_STATE_HUMAN,
    )



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, label_map_list: list, ct_map_list: list = None):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot_US"]
    human = scene['human']
    robot_entity_cfg = SceneEntityCfg("robot_US", joint_names=["lbr_joint_.*"], body_names=["lbr_link_ee"])
    robot_entity_cfg.resolve(scene)
    US_ee_jacobi_idx = robot_entity_cfg.body_ids[-1]

    # define ik controllers
    ik_params = {"lambda_val": 0.00001}
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls", ik_params=ik_params)
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, scene.num_envs, device=sim.device)
    pose_diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls", ik_params=ik_params)
    pose_diff_ik_controller = DifferentialIKController(pose_diff_ik_cfg, scene.num_envs, device=sim.device)

    # construct label image slicer
    label_convert_map = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/label_conversion.yaml", 'r'))

    # construct US simulator
    us_cfg = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_cfg.yaml", 'r'))
    sim_cfg = scene_cfg['sim']
    US_slicer = USSlicer(
        us_cfg,
        label_map_list, 
        ct_map_list,
        sim_cfg['if_use_ct'],
        human_stl_list,
        scene.num_envs, 
        sim_cfg['patient_xz_range'], 
        sim_cfg['patient_xz_init'], 
        sim.device, 
        label_convert_map,
        us_cfg['image_size'], 
        us_cfg['resolution'],
        visualize=sim_cfg['vis_seg_map'],
    )

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running() and count < 1000:
        # Reset
        if count % sim_cfg['episode_length'] == 0:
            # reset counter
            # reset the scene entities
            # root state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

            diff_ik_controller.reset()
            pose_diff_ik_controller.reset()

            # get ee pose in base frame
            US_root_pose_w = robot.data.root_state_w[:, 0:7]

            US_ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[-1], 0:7]
            # compute frame in root frame
            US_ee_pos_b, US_ee_quat_b = subtract_frame_transforms(
                US_root_pose_w[:, 0:3], US_root_pose_w[:, 3:7], US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7]
            )

            ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=sim.device)
            diff_ik_controller.set_command(ik_commands, US_ee_pos_b, US_ee_quat_b)
            ik_commands_pose = torch.zeros(scene.num_envs, pose_diff_ik_controller.action_dim, device=sim.device)
            pose_diff_ik_controller.set_command(ik_commands_pose, US_ee_pos_b, US_ee_quat_b)

            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")

        start = time.time()
        # Apply random action
        rand_x_z_angle = torch.rand((scene.num_envs, 3), device=sim.device) * 2.0 - 1.0
        rand_x_z_angle[:, 2] = (rand_x_z_angle[:, 2] / 10)
        US_slicer.update_cmd(rand_x_z_angle)

        # get human frame
        human_world_poses = human.data.root_state_w # these are already the initial poses
        # define world to human poses
        world_to_human_pos, world_to_human_rot = human_world_poses[:, 0:3], human_world_poses[:, 3:7]

        # update the view
        # get ee pose in wolrd frame
        US_ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[-1], 0:7]
        world_to_base_pose = robot.data.root_link_state_w[:, 0:7]
        US_ee_pos_b, US_ee_quat_b = subtract_frame_transforms(
            world_to_base_pose[:, 0:3], world_to_base_pose[:, 3:7], US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7]
        )
        # update image simulation
        US_slicer.slice_US(world_to_human_pos, world_to_human_rot, US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7])
        if sim_cfg['vis_us']:
            US_slicer.visualize()
        
        # compute frame in root frame
        if sim_cfg['vis_seg_map']:
            US_slicer.update_plotter(world_to_human_pos, world_to_human_rot, US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7])
        world_to_ee_target_pos, world_to_ee_target_rot = US_slicer.compute_world_ee_pose_from_cmd(
            world_to_human_pos, world_to_human_rot)
        world_to_ee_target_pose = torch.cat([world_to_ee_target_pos, world_to_ee_target_rot], dim=-1)
        
        base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
            world_to_base_pose[:, 0:3], world_to_base_pose[:, 3:7], world_to_ee_target_pos, world_to_ee_target_rot
        )
        base_to_ee_target_pose = torch.cat([base_to_ee_target_pos, base_to_ee_target_quat], dim=-1)

        # set new command
        pose_diff_ik_controller.set_command(base_to_ee_target_pose)

        # torch.cuda.synchronize()
        # # get joint position targets
        US_jacobian = robot.root_physx_view.get_jacobians()[:, US_ee_jacobi_idx-1, :, robot_entity_cfg.joint_ids]
        US_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        # compute the joint commands
        joint_pos_des = pose_diff_ik_controller.compute(
            US_ee_pos_b, 
            US_ee_quat_b,
            US_jacobian, 
            US_joint_pos
        )
        
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)

        end = time.time()
        print(f"Time taken for step: {end - start}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device) # , gravity=[0.0, 0.0, 0.0]
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    robot_scene_cfg = RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=False)
    scene = InteractiveScene(robot_scene_cfg)
    # load label maps
    label_map_list = []
    for label_map_file in label_map_file_list:
        label_map = nib.load(label_map_file).get_fdata()
        label_map_list.append(label_map)
    # load ct maps
    ct_map_list = []
    for ct_map_file in ct_map_file_list:
        ct_map = nib.load(ct_map_file).get_fdata()
        ct_min_max = scene_cfg['sim']['ct_range']
        ct_map = np.clip(ct_map, ct_min_max[0], ct_min_max[1])
        ct_map = (ct_map - ct_min_max[0]) / (ct_min_max[1] - ct_min_max[0]) * 255
        ct_map_list.append(ct_map)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene, label_map_list, ct_map_list)


if __name__ == "__main__":
    # run the main function
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("main_stats.prof")

    # close sim app
    simulation_app.close()