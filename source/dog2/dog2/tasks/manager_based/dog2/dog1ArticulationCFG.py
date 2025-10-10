# Create a new file: ~/dog1/dog1/tasks/manager_based/dog1/assets/my_dog_cfg.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration for your custom dog robot
##

# MY_DOG_ACTUATOR_CFG = DCMotorCfg(
#     joint_names_expr=[".*"],  # Will match all joints in your URDF
#     saturation_effort=120.0,  # Adjust based on your servo specs
#     effort_limit=80.0,       # 80% safety margin
#     velocity_limit=17.5,      # Conservative speed limit (rad/s)
#     stiffness={".*": 100.0},   # Start moderate
#     damping={".*": 1.0},      # Start low
# )

# MY_DOG_CFG = ArticulationCfg(
#     prim_path="/World/envs/env_.*/Robot",   # <— add this
#     spawn=sim_utils.UsdFileCfg(
#         usd_path="/home/chenry/Desktop/legged_software/legged_descriptions/go2_description/urdf/go2_simplified/go2_simplified.usd",  # Path to your converted USD
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True,  
#             solver_position_iteration_count=4, 
#             solver_velocity_iteration_count=0
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.40),  # Adjust height based on your dog's leg length
#         joint_pos={
#             # Front left leg
#             "FL_hip_joint": 0.0,         # 0° (matches inspector)
#             "FL_thigh_joint": 0.7854,    # 45° in radians (matches inspector)
#             "FL_calf_joint": -1.5708,    # -90° in radians (matches inspector)
            
#             # Front right leg  
#             "FR_hip_joint": 0.0,         # 0° (matches inspector)
#             "FR_thigh_joint": 0.7854,    # 45° in radians (matches inspector)
#             "FR_calf_joint": -1.5708,    # -90° in radians (matches inspector)
            
#             # Rear left leg
#             "RL_hip_joint": 0.0,         # 0° (matches inspector)
#             "RL_thigh_joint": 0.7854,    # 45° in radians (matches inspector)
#             "RL_calf_joint": -1.5708,    # -90° in radians (matches inspector)
            
#             # Rear right leg
#             "RR_hip_joint": 0.0,         # 0° (matches inspector)
#             "RR_thigh_joint": 0.7854,    # 45° in radians (matches inspector)
#             "RR_calf_joint": -1.5708,    # -90° in radians (matches inspector)
#         },
#     ),
#     actuators={"legs": MY_DOG_ACTUATOR_CFG},
#     soft_joint_pos_limit_factor=0.95,
# )

MY_DOG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
        usd_path="/home/chenry/Desktop/legged_software/legged_descriptions/go2_description/urdf/go2_simplified/go2_simplified.usd",
        activate_contact_sensors=True,
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)