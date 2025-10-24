#!/usr/bin/env python3

"""Simple script to test raycasting in Isaac Sim."""

import torch
import argparse
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Test raycasting")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import SceneEntityCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# Import your robot config
try:
    from source.dog2.dog2.tasks.manager_based.dog2.dog1ArticulationCFG import MY_DOG_CFG
except ImportError:
    print("Warning: Could not import dog config, using default")
    from isaaclab_assets.robots.cartpole import CARTPOLE_CFG as MY_DOG_CFG


@configclass
class RaycastTestSceneCfg(InteractiveSceneCfg):
    """Simple scene for raycasting test."""
    
    # Ground
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    
    # Robot
    robot: ArticulationCfg = MY_DOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # Height scanner with VERY visible rays
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 2.0)),  # Lower height for better visualization
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[2.0, 1.0]),  # Bigger, fewer rays
        debug_vis=True,  # Enable visualization
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class SimpleActionsCfg:
    """Minimal actions configuration."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class SimpleObservationsCfg:
    """Minimal observations configuration."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        height_scan = ObservationTermCfg(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass 
class RaycastTestEnvCfg(ManagerBasedEnvCfg):
    """Simple environment for raycasting test."""
    
    scene: RaycastTestSceneCfg = RaycastTestSceneCfg(num_envs=1, env_spacing=4.0)
    observations: SimpleObservationsCfg = SimpleObservationsCfg()
    actions: SimpleActionsCfg = SimpleActionsCfg()
    
    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        
        # Update sensor period
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt


def main():
    """Run the raycasting test."""
    print("🔍 Starting raycasting test...")
    
    # Create environment
    env_cfg = RaycastTestEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)
    
    print(f"✅ Environment created with {env.num_envs} environment(s)")
    print(f"🎯 Height scanner: {env.scene.sensors.keys()}")
    
    # Reset environment
    env.reset()
    
    print("🚀 Running simulation...")
    print("👁️  Look in Isaac Sim viewport for green/red rays!")
    print("   - Green rays: Hit ground")
    print("   - Red rays: No collision")
    print("   - Press Ctrl+C to stop")
    
    step_count = 0
    try:
        while simulation_app.is_running():
            # Random actions to make robot move
            actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            
            # Step simulation
            env.step(actions)
            
            # Print height data occasionally
            if step_count % 100 == 0:
                sensor = env.scene.sensors["height_scanner"]
                height_data = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2]
                
                print(f"\n📊 Step {step_count}: Height scan stats")
                print(f"   Shape: {height_data.shape}")
                print(f"   Min: {height_data.min().item():.3f}m")
                print(f"   Max: {height_data.max().item():.3f}m")
                print(f"   Mean: {height_data.mean().item():.3f}m")
            
            step_count += 1
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping simulation...")
    
    # Clean up
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()