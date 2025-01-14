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
parser.add_argument("--num_envs", type=int, default=200, help="Number of environments to spawn.")
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

##
# Pre-defined configs
##
from spinal_surgery.assets.kuka_US import *
from omni.isaac.lab.utils.math import subtract_frame_transforms, combine_frame_transforms
from pxr import Gf, UsdGeom
from scipy.spatial.transform import Rotation as R
from spinal_surgery.lab.kinematics.human_frame_viewer import HumanFrameViewer
from spinal_surgery.lab.kinematics.surface_motion_planner import SurfaceMotionPlanner
from spinal_surgery.lab.kinematics.label_img_slicer import LabelImgSlicer

INIT_STATE_ROBOT_US = ArticulationCfg.InitialStateCfg(
    joint_pos={
        "lbr_joint_0": 1.5,
        "lbr_joint_1": -0.1,
        "lbr_joint_2": 0.0,
        "lbr_joint_3": -1.6, # -1.2,
        "lbr_joint_4": 0.0,
        "lbr_joint_5": 1.6, # 1.5,
        "lbr_joint_6": 0.0,
    },
    pos = (0.0, -0.75, 0.2)
)

quat = R.from_euler("yxz", (-90, -90, 0), degrees=True).as_quat()
INIT_STATE_HUMAN = AssetBaseCfg.InitialStateCfg(
    pos=((0.2, -0.4, 0.6)),
    rot=((quat[3], quat[0], quat[1], quat[2]))
)

quat = R.from_euler("xyz", (90, 0, 90), degrees=True).as_quat()
INIT_STATE_BED = AssetBaseCfg.InitialStateCfg(
    pos=((0.0, 0.0, 0.0)),
    rot=((quat[3], quat[0], quat[1], quat[2]))
)

human_usd_list = [
            f"{ASSETS_DATA_DIR}/HumanModels/Totalsegmentator_dataset_v2_subset_usd_no_col/s0010", 
            f"{ASSETS_DATA_DIR}/HumanModels/Totalsegmentator_dataset_v2_subset_usd_no_col/s0014",
            f"{ASSETS_DATA_DIR}/HumanModels/Totalsegmentator_dataset_v2_subset_usd_no_col/s0015",
]
human_stl_list = [
            f"{ASSETS_DATA_DIR}/HumanModels/Totalsegmentator_dataset_v2_subset_stl/s0010", 
            f"{ASSETS_DATA_DIR}/HumanModels/Totalsegmentator_dataset_v2_subset_stl/s0014",
            f"{ASSETS_DATA_DIR}/HumanModels/Totalsegmentator_dataset_v2_subset_stl/s0015",
]

usd_file_list = [human_file + "/combined/combined.usd" for human_file in human_usd_list]
label_map_file_list = [human_file + "/combined_label_map.nii.gz" for human_file in human_stl_list]

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
            scale = (0.001, 0.001, 0.001),
        ),
        init_state = INIT_STATE_BED
    )


    # human: 
    # human = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Human", 
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ASSETS_DATA_DIR}/HumanModels/Totalsegmentator_dataset_v2_subset_usd_no_col/s0021/combined/combined.usd",
    #         scale = (0.001, 0.001, 0.001),
    #         # translation = (0.0, 0.0, 1.05), # to make the asset static by specifying the translation
    #         # orientation = (1.0, 0.0, 0.0, 0.0)
    #     ),
    #     init_state = INIT_STATE_HUMAN
    # )

    human = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Human", 
        spawn=sim_utils.MultiUsdFileCfg(
        usd_path=usd_file_list,
        random_choice=False,
        scale = (0.001, 0.001, 0.001),
        ),
        init_state = INIT_STATE_HUMAN,
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, label_map_list: list):
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

    # construct the human frame viewer:
    # human_frame_viewer = HumanFrameViewer(label_map_list, scene.num_envs)
    # surface_motion_planner = SurfaceMotionPlanner(
    #     label_map_list, 
    #     scene.num_envs, 
    #     [[50, 50, 2.0], [150, 200, 4.0]], [100, 120, 3.14], 
    #     sim.device)
    label_img_slicer = LabelImgSlicer(
        label_map_list, 
        human_stl_list,
        scene.num_envs, 
        [[100, 100, 2.64], [200, 200, 3.64]], [150, 150, 3.14], 
        sim.device, [150, 200], 0.0003
    )

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
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

            # get data from the human
            human_root_state = human.get_default_state()
            human_local_pose = human.get_local_poses()
            human_world_poses = human.get_world_poses() # these are already the initial poses

            # define world to human poses
            world_to_human_pos, world_to_human_rot = human_root_state.positions, human_root_state.orientations

            # then we can compute the relative poses and conduct simulations

            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply random action
        rand_x_z_angle = torch.rand((scene.num_envs, 3), device=sim.device) * 2.0 - 1.0
        rand_x_z_angle[:, 2] = (rand_x_z_angle[:, 2] / 10)
        label_img_slicer.update_cmd(rand_x_z_angle)

        # update the view
        # get ee pose in wolrd frame
        US_ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[-1], 0:7]
        world_to_base_pose = robot.data.root_link_state_w[:, 0:7]
        US_ee_pos_b, US_ee_quat_b = subtract_frame_transforms(
            world_to_base_pose[:, 0:3], world_to_base_pose[:, 3:7], US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7]
        )
        # update image simulation
        label_img_slicer.slice_label_img(world_to_human_pos, world_to_human_rot, US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7])
        label_img_slicer.visualize()
        
        # compute frame in root frame
        label_img_slicer.update_plotter(world_to_human_pos, world_to_human_rot, US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7])
        world_to_ee_target_pos, world_to_ee_target_rot = label_img_slicer.compute_world_ee_pose_from_cmd(world_to_human_pos, world_to_human_rot)
        world_to_ee_target_pose = torch.cat([world_to_ee_target_pos, world_to_ee_target_rot], dim=-1)
        
        base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
            world_to_base_pose[:, 0:3], world_to_base_pose[:, 3:7], world_to_ee_target_pos, world_to_ee_target_rot
        )
        base_to_ee_target_pose = torch.cat([base_to_ee_target_pos, base_to_ee_target_quat], dim=-1)

        # set new command
        pose_diff_ik_controller.set_command(base_to_ee_target_pose)
        
        # get joint position targets
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


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, gravity=[0.0, 0.0, 0.0])
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0, replicate_physics=False)
    scene = InteractiveScene(scene_cfg)
    # load label maps
    label_map_list = []
    for label_map_file in label_map_file_list:
        label_map = nib.load(label_map_file).get_fdata()
        label_map_list.append(label_map)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene, label_map_list)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()