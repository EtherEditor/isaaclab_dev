"""
Phase 3: Domain Randomization & Events (Strictly Vectorized)
Filename: event_cfg.py
Innovation 1 addition: SDF rebuild triggered after obstacle reset.
"""
import torch
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
from sdf_guidance import rebuild_sdf as _rebuild_sdf

# -------------------------------------------------------------------------
# Custom Vectorized MDP Functions
# -------------------------------------------------------------------------
# Track which obstacles have already emitted a missing-entity warning
# to prevent log spam (the event fires every reset, i.e. thousands of times).
_obstacle_warned: set[str] = set()


def reset_obstacles_curriculum_safe(
    env,
    env_ids: torch.Tensor,
    obstacle_names: list[str], # Use explicit string keys to bypass the resolver
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    spacing_bounds: tuple[float, float],
    decay_steps: int
):
    num_envs_to_reset = len(env_ids)
    num_obstacles = len(obstacle_names)

    # 1. Compute dynamic spacing d(t)
    progress = min(1.0, env.common_step_counter / decay_steps)
    current_spacing = spacing_bounds[1] - progress * (spacing_bounds[1] - spacing_bounds[0])

    y_centers = torch.linspace(y_range[0], y_range[1], num_obstacles, device=env.device)

    # 2. Iterate and fetch tensor views directly by name
    for i, name in enumerate(obstacle_names):
        try:
            asset = env.scene[name]
        except KeyError:
            # Only warn once per entity name to avoid flooding the log.
            if name not in _obstacle_warned:
                print(f"[WARN] Obstacle entity '{name}' not found in scene; skipping placement.")
                _obstacle_warned.add(name)
            continue

        positions = torch.zeros((num_envs_to_reset, 3), device=env.device)
        positions[:, 0] = torch.empty(num_envs_to_reset, device=env.device).uniform_(*x_range)

        y_noise = torch.empty(num_envs_to_reset, device=env.device).uniform_(-current_spacing/4, current_spacing/4)
        positions[:, 1] = y_centers[i] + y_noise
        positions[:, 2] = 0.25

        asset.set_world_poses(positions, env_indices=env_ids)
# -----------------------------------------------------------------
    # Innovation 1: rebuild SDF after placing obstacles
    # This is a no-op when ConsistencyArmPolicy is not in the action
    # manager, preserving full backward compatibility.
    # -----------------------------------------------------------------
    _rebuild_sdf(env, env_ids)

def _reset_visual_delay(env, env_ids: torch.Tensor) -> None:
    """Re-sample visual delay and clear the image buffer for reset environments."""
    if hasattr(env, "_visual_buffer") and env._visual_buffer is not None:
        env._visual_buffer.reset(env_ids)

# -------------------------------------------------------------------------
# Event Configuration
# -------------------------------------------------------------------------
@configclass
class Go2RetrievalEventCfg:
    """Highly scalable domain randomization and procedural resets."""

    # -------------------------------------------------------------------------
    # 1. Startup: Domain Randomization
    # -------------------------------------------------------------------------
    randomize_terrain_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.01), # Minimal bounce
            "num_buckets": 250, # Required: Discretizes the distribution
        },
    )

    # =========================================================================
    # Innovation 3: visual delay sampling and buffer reset
    # =========================================================================
    reset_visual_delay = EventTerm(
        func=_reset_visual_delay,
        mode="reset",
        params={},
    )

    # -------------------------------------------------------------------------
    # Update Startup: Explicit Mass Randomization
    # -------------------------------------------------------------------------
    randomize_obs_0_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "mass_distribution_params": (5.0, 15.0), "operation": "scale"}
    )
    randomize_obs_1_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "mass_distribution_params": (5.0, 15.0), "operation": "scale"}
    )
    randomize_obs_2_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "mass_distribution_params": (5.0, 15.0), "operation": "scale"}
    )

    # -------------------------------------------------------------------------
    # Update Resets: List Comprehension for Assets
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # 2. Resets: State & Vectorized Spatial Logic
    # -------------------------------------------------------------------------
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.1, 0.1)},
            "velocity_range": {},
        },
    )

    # Vectorized, collision-free placement of the obstacle cluster
    reset_obstacles_safe = EventTerm(
        func=reset_obstacles_curriculum_safe,
        mode="reset",
        params={
            # Pass the list of entities to your refactored function
            "obstacle_names": ["obstacle_0", "obstacle_1", "obstacle_2"],
            "x_range": (1.0, 3.0),
            "y_range": (-2.0, 2.0),
            "spacing_bounds": (0.4, 1.5),
            "decay_steps": 5_000_000,
        },
    )

    # Target placement mathematically constrained behind the obstacle zone
    reset_target_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("target_object"),
            "pose_range": {"x": (3.5, 5.0), "y": (-2.0, 2.0)},
            "velocity_range": {},
        },
    )

    # -------------------------------------------------------------------------
    # 3. Interval: Robustness via Force/Torque Wrenches
    # -------------------------------------------------------------------------

    push_robot_wrench = EventTerm(
        # Standard IsaacLab MDP formulation for applying physical wrenches
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            # sample_uniform expects a flat (min, max) tuple, not a per-axis dict
            "force_range": (-30.0, 30.0),
            "torque_range": (-5.0, 5.0),
        },
    )