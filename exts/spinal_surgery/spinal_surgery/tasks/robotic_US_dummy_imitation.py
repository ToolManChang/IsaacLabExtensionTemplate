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
from spinal_surgery.lab.kinematics.gt_motion_generator import GTMotionGenerator
from spinal_surgery.lab.agents.imitation_agents import ImitationAgent, PositionAgent
import torch.optim as optim
import torch.nn.functional as F
from ruamel.yaml import YAML
from spinal_surgery import PACKAGE_DIR


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
    pos = (0.0, -0.7, 0.7) # ((0.0, -0.75, 0.4))
)

quat = R.from_euler("yxz", (-90, -90, 0), degrees=True).as_quat()
INIT_STATE_HUMAN = RigidObjectCfg.InitialStateCfg(
    pos=((0.2, -0.45, 1.1)), # 0.7
    rot=((quat[3], quat[0], quat[1], quat[2]))
)

quat = R.from_euler("xyz", (90, 0, 90), degrees=True).as_quat()
INIT_STATE_BED = AssetBaseCfg.InitialStateCfg(
    pos=((0.0, 0.0, 0.3)),
    rot=((quat[3], quat[0], quat[1], quat[2]))
)
# use stl: Totalsegmentator_dataset_v2_subset_body_contact
human_usd_list = [
            f"{ASSETS_DATA_DIR}/HumanModels/Totalsegmentator_dataset_v2_subset_body_from_urdf/s0010", 
            # f"{ASSETS_DATA_DIR}/HumanModels/Totalsegmentator_dataset_v2_subset_body_from_urdf/s0014",
            # f"{ASSETS_DATA_DIR}/HumanModels/Totalsegmentator_dataset_v2_subset_body_from_urdf/s0015",
]
human_stl_list = [
            f"{ASSETS_DATA_DIR}/HumanModels/Totalsegmentator_dataset_v2_subset_stl/s0010", 
            # f"{ASSETS_DATA_DIR}/HumanModels/Totalsegmentator_dataset_v2_subset_stl/s0014",
            # f"{ASSETS_DATA_DIR}/HumanModels/Totalsegmentator_dataset_v2_subset_stl/s0015",
]
usd_file_list = [human_file + "/combined_wrapwrap/combined_wrapwrap.usd" for human_file in human_usd_list]
label_map_file_list = [human_file + "/combined_label_map.nii.gz" for human_file in human_stl_list]

label_res = 0.0015
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
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
           articulation_enabled=False,
        ),
        ),
        init_state = INIT_STATE_HUMAN,
    )

    # human_collision = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Human_collision",
    #     spawn=sim_utils.MultiUsdFileCfg(
    #         usd_path=collision_file_list,
    #         random_choice=False,
    #         scale = (label_res, label_res, label_res),
    #     ),
    #     init_state = INIT_STATE_HUMAN
    # )


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

    # construct label image slicer
    label_convert_map = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/label_conversion.yaml", 'r'))
   
    init_cmd_pose = torch.tensor([[150, 180, 2.5]], device=sim.device).reshape((1, -1)).repeat(scene.num_envs, 1)
    goal_cmd_pose = torch.tensor([[170, 180, 2.0]], device=sim.device).reshape((1, -1)).repeat(scene.num_envs, 1)

    # construct US simulator
    us_cfg = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_cfg.yaml", 'r'))
    US_slicer = USSlicer(
        us_cfg,
        label_map_list, 
        human_stl_list,
        scene.num_envs, 
        [[100, 50, 1.5], [200, 230, 3.14]], [150, 180, 2.5], 
        sim.device, label_convert_map,
        [150, 200], 0.0004, visualize=False
    )
    gt_motion_generator = GTMotionGenerator(
        goal_cmd_pose=goal_cmd_pose,
        scale=5,
        num_envs=scene.num_envs,
        surface_map_list=US_slicer.surface_map_list,
        surface_normal_list=US_slicer.surface_normal_list,
        label_res=label_res,
        US_height=US_slicer.height,
    )

    # learning
    # cfg = YAML().load(open(f"{PACKAGE_DIR}/lab/agents/cfgs/imitation_agent_cfg.yaml", 'r'))
    # agent = ImitationAgent(cfg, scene.num_envs, sim.device, US_slicer.img_size)
    cfg = YAML().load(open(f"{PACKAGE_DIR}/lab/agents/cfgs/position_agent_cfg.yaml", 'r'))
    agent = PositionAgent(cfg, scene.num_envs, sim.device)
    

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    episode = 0
    # Simulation loop
    while simulation_app.is_running() and episode <= 200:
        # Reset
        if count % 500 == 0:
            episode += 1
            # reset counter
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
            # human_root_state = human.get_default_state() # for assets
            # human_local_pose = human.get_local_poses()
            

            # then we can compute the relative poses and conduct simulations

            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")

            US_slicer.update_cmd(init_cmd_pose - US_slicer.current_x_z_x_angle_cmd)

            # train
            # print('GT Command:', gt_cmd[:5,:])
           
            # print(output.shape, gt_output.shape)
            if count>0 and not episode % 5==1:

                agent.train()

            agent.reset()
            # hidden = torch.zeros((1, scene.num_envs, agent.hidden_size), device=sim.device)
            # last_vector = torch.zeros((scene.num_envs, 1, 7), device=sim.device)



        start = time.time()

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
        # US_slicer.visualize()

        # go to the init pose
        if count % 500 < 100:
            # Apply random action # cover tha init state
            if not episode % 5==0:
                rand_scale = torch.tensor((100, 50, 1.64), device=sim.device).reshape((1, -1))
                rand_start = torch.tensor((100, 50, 1.5), device=sim.device).reshape((1, -1))
                rand_x_z_angle = torch.rand((scene.num_envs, 3), device=sim.device)*rand_scale + rand_start
            else:
                rand_scale = torch.tensor((60, 30, 1.5), device=sim.device).reshape((1, -1))
                rand_start = torch.tensor((120, 60, 1.7), device=sim.device).reshape((1, -1))
                rand_x_z_angle = torch.rand((scene.num_envs, 3), device=sim.device)*rand_scale + rand_start

            # rand_x_z_angle = torch.zeros((scene.num_envs, 3), device=sim.device)
            US_slicer.update_cmd(rand_x_z_angle-US_slicer.current_x_z_x_angle_cmd)
        else:
            # get cur command
            us_imgs = US_slicer.us_img_tensor.unsqueeze(1).unsqueeze(1).float() / 255.0 # (num_envs, 1, 1, w, h)

            # output, next_hidden = agent.predict(images=us_imgs, vectors=last_vector, hidden_states=hidden)
            cur_human_ee_pos, cur_human_ee_quat = subtract_frame_transforms(
            world_to_human_pos, world_to_human_rot, US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7]
            )

            # convert current pose to human command pose
            cur_cmd_pose = gt_motion_generator.human_cmd_state_from_ee_pose(cur_human_ee_pos, cur_human_ee_quat)
            cur_cmd_pose /= 100
            output = agent.predict(cur_cmd_pose)
            
            out_quat = output[:, 3:7] / (output[:, 3:7].norm(dim=-1, keepdim=True) + 1e-9)
            cur_cmd = gt_motion_generator.compute_human_cmd_from_current_ee_cmd(
                US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7], output[:, 0:3].clone().detach()/1000, out_quat.clone().detach(), 
                world_to_human_pos, world_to_human_rot
            )
            # train with gt command
            gt_ee_target_pos, gt_ee_target_quat = gt_motion_generator.generate_gt_ee_cmd_from_current_pose(
                US_slicer, US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7], world_to_human_pos, world_to_human_rot
            )
            gt_output = torch.cat([gt_ee_target_pos, gt_ee_target_quat], dim=-1) # (num_envs, 7)
            gt_cmd = gt_motion_generator.compute_human_cmd_from_current_ee_cmd(
                US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7], gt_ee_target_pos, gt_ee_target_quat, 
                world_to_human_pos, world_to_human_rot
            )
            # print('GT Command:', gt_cmd)
            print('episode', episode)
            # print('gt_cmd', gt_cmd[:5,:])

            # update buffer
            # agent.record(images=us_imgs, vectors=last_vector, hidden_states=hidden, gt_output=gt_output)
            agent.record(cur_cmd_pose, gt_output)

            diff_cmd = cur_cmd-US_slicer.current_x_z_x_angle_cmd
            
            if episode % 5==0:
                # diff_cmd = diff_cmd + rand_x_z_angle
                US_slicer.update_cmd(diff_cmd)
            # Apply random action
            else: 
                # random
                # rand_scale = torch.tensor((100, 180, 1.64), device=sim.device).reshape((1, -1))
                # rand_start = torch.tensor((100, 50, 1.5), device=sim.device).reshape((1, -1))
                # rand_x_z_angle = torch.rand((scene.num_envs, 3), device=sim.device)*rand_scale + rand_start
                # US_slicer.update_cmd(rand_x_z_angle-US_slicer.current_x_z_x_angle_cmd)
                # gt
                diff_cmd = gt_cmd-US_slicer.current_x_z_x_angle_cmd
                rand_x_z_angle = torch.rand((scene.num_envs, 3), device=sim.device) * 40.0 - 20.0
                rand_x_z_angle[:, 2] = (rand_x_z_angle[:, 2] / 10)
                diff_cmd = diff_cmd + rand_x_z_angle
                US_slicer.update_cmd(diff_cmd)
            
        
        # compute frame in root frame
        # US_slicer.update_plotter(world_to_human_pos, world_to_human_rot, US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7])
        world_to_ee_target_pos, world_to_ee_target_rot = US_slicer.compute_world_ee_pose_from_cmd(
            world_to_human_pos, world_to_human_rot)
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

        end = time.time()
        # print(f"Time taken for step: {end - start}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device) # , gravity=[0.0, 0.0, 0.0]
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=False)
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
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("main_stats.prof")

    # close sim app
    simulation_app.close()