"""
MDP Configuration (Stages 2–3, consolidated)
Filename: mdp_cfg.py
"""
import torch
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp import RelativeJointPositionActionCfg
from custom_locomotion import HierarchicalLocomotionAction, HierarchicalLocomotionActionCfg


# =========================================================================
# Section 1: All raw mathematical functions
# These must all be defined before any @configclass that references them.
# =========================================================================

def lidar_distance(env, sensor_name: str) -> torch.Tensor:
    """
    Retrieves ray distances from a RayCasterCamera sensor.
    Returns shape (num_envs, num_rays).
    """
    sensor = env.scene.sensors[sensor_name]
    # RayCasterData provides hit positions (`ray_hits_w`) and sensor origin (`pos_w`).
    # Compute per-ray distance as ||hit_pos - sensor_pos||. Shape: (N, B).
    ray_hits = sensor.data.ray_hits_w  # (N, B, 3)
    sensor_pos = sensor.data.pos_w     # (N, 3)
    if ray_hits is None or sensor_pos is None:
        # Return an empty tensor if sensor data not populated yet
        return torch.zeros((0, 0), device=env.device)
    d = torch.norm(ray_hits - sensor_pos.unsqueeze(1), dim=-1)  # (N, B)
    return d.view(d.shape[0], -1)


def custom_obstacle_proximity_penalty(
    env, sensor_name: str, threshold: float, sigma: float
) -> torch.Tensor:
    """
    Gaussian obstacle proximity penalty for use in reward shaping.

    r_obs = -w * Σ_i exp(-d_i² / σ²)  for d_i < threshold
    """
    ray_distances = env.scene.sensors[sensor_name].data.ray_hits_w
    sensor_pos = env.scene.sensors[sensor_name].data.pos_w
    if ray_distances is None or sensor_pos is None:
        return torch.zeros((0,), device=env.device)
    ray_distances = torch.norm(ray_distances - sensor_pos.unsqueeze(1), dim=-1)
    mask    = ray_distances < threshold
    penalty = torch.zeros_like(ray_distances)
    penalty[mask] = torch.exp(-(ray_distances[mask] ** 2) / (sigma ** 2))
    return torch.sum(penalty, dim=-1)


def piecewise_retrieval_reward(
    env,
    target_name: str,
    home_pos: tuple[float, float, float],
    sigma: float,
) -> torch.Tensor:
    """
    Two-stage Gaussian shaping reward.

    R = (1 - I) * exp(-d_target² / σ²) + I * exp(-d_home² / σ²)
    """
    robot_pos   = env.scene["robot"].data.root_pos_w
    target_pos  = env.scene[target_name].data.root_pos_w
    home_tensor = torch.tensor(home_pos, device=env.device)

    dist_target = torch.norm(target_pos - robot_pos, dim=-1)
    dist_home   = torch.norm(home_tensor - robot_pos, dim=-1)
    I_payload   = env.has_payload.float()

    r_app = torch.exp(-(dist_target ** 2) / (sigma ** 2))
    r_ret = torch.exp(-(dist_home ** 2) / (sigma ** 2))
    return (1.0 - I_payload) * r_app + I_payload * r_ret


def velocity_projection_reward(env, target_name: str) -> torch.Tensor:
    """
    Projects robot velocity onto the unit vector toward the target.

    R_vel = v_robot · (p_target - p_robot) / ||p_target - p_robot||
    """
    robot_pos  = env.scene["robot"].data.root_pos_w
    target_pos = env.scene[target_name].data.root_pos_w
    v_robot    = env.scene["robot"].data.root_lin_vel_w

    direction      = target_pos - robot_pos
    direction_norm = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-5)
    return torch.sum(v_robot * direction_norm, dim=-1)


def ee_reaching_reward(
    env,
    ee_frame_name: str,
    target_name: str,
    sigma: float,
) -> torch.Tensor:
    """
    Gaussian end-effector reaching reward.

    R_reach = exp(-||p_ee - p_target||² / σ²)
    """
    ee_sensor = env.scene.sensors[ee_frame_name]
    p_ee      = ee_sensor.data.target_pos_w[:, 0, :]
    p_target  = env.scene[target_name].data.root_pos_w
    dist_sq   = torch.sum((p_ee - p_target) ** 2, dim=-1)
    return torch.exp(-dist_sq / (sigma ** 2))


def arm_collision_cost(
    env,
    sensor_name: str,
    force_threshold: float,
) -> torch.Tensor:
    """
    Binary cost: 1.0 if any Z1 link contact force exceeds the threshold.

    C_arm = 1{max_body ||F||₂ > F_threshold}
    """
    sensor         = env.scene.sensors[sensor_name]
    net_forces     = sensor.data.net_forces_w                       # (N, B, 3)
    force_mags     = torch.norm(net_forces, dim=-1)                 # (N, B)
    max_force      = torch.max(force_mags, dim=-1).values           # (N,)
    return (max_force > force_threshold).float()


def base_proximity_cost(
    env,
    sensor_name: str,
    d_safe: float,
) -> torch.Tensor:
    """
    Continuous hinge-loss proximity cost, normalised by ray count.

    C_prox = (1 / N_rays) · Σ_i max(0, d_safe - d_i)

    Normalising by ray count makes the budget κ independent of LiDAR
    resolution, so the same threshold remains meaningful if horizontal_res
    is changed in scene_cfg.py.
    """
    sensor        = env.scene.sensors[sensor_name]
    ray_hits      = sensor.data.ray_hits_w
    sensor_pos    = sensor.data.pos_w
    if ray_hits is None or sensor_pos is None:
        return torch.zeros((0,), device=env.device)
    ray_distances = torch.norm(ray_hits - sensor_pos.unsqueeze(1), dim=-1)
    violation     = torch.clamp(d_safe - ray_distances, min=0.0)
    num_rays      = ray_distances.shape[-1] if ray_distances.ndim > 1 else 1
    return violation.sum(dim=-1) / num_rays


def task_completed(
    env,
    home_pos: tuple[float, float, float],
    threshold: float,
) -> torch.Tensor:
    """Termination condition: payload retrieved and robot returned to origin."""
    home_tensor = torch.tensor(home_pos, device=env.device)
    robot_pos   = env.scene["robot"].data.root_pos_w
    dist_home   = torch.norm(home_tensor - robot_pos, dim=-1)
    return env.has_payload & (dist_home < threshold)


# =========================================================================
# Section 2: Observation configurations
# =========================================================================

@configclass
class Go2ExteroceptionObsCfg(ObsGroup):
    target_pos    = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "target_pose"},
    )
    obstacle_scan = ObsTerm(
        func=lidar_distance,
        params={"sensor_name": "lidar"},
    )


@configclass
class Go2ProprioceptionObsCfg(ObsGroup):
    """Internal kinematic state vector."""
    base_lin_vel    = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel    = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    joint_pos_err   = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel       = ObsTerm(func=mdp.joint_vel_rel)
    last_action     = ObsTerm(func=mdp.last_action)
    has_payload     = ObsTerm(func=lambda env: env.has_payload.view(-1, 1).float())


# =========================================================================
# Section 3: Reward and cost configurations
# All functions referenced here are defined in Section 1.
# =========================================================================

@configclass
class ArmedGo2RewardCfg:
    """
    Task reward configuration for the Go2 + Z1 mobile manipulation MDP.
    Obstacle avoidance is expressed exclusively as a safety cost in
    ArmedGo2CostCfg, in accordance with the CMDP separation principle.
    """
    forward_momentum: RewTerm = RewTerm(
        func=velocity_projection_reward,
        weight=0.5,
        params={"target_name": "target_object"},
    )
    ee_reaching: RewTerm = RewTerm(
        func=ee_reaching_reward,
        weight=3.0,
        params={
            "ee_frame_name": "ee_frame",
            "target_name":   "target_object",
            "sigma":         0.3,
        },
    )
    task_progress: RewTerm = RewTerm(
        func=piecewise_retrieval_reward,
        weight=2.0,
        params={
            "target_name": "target_object",
            "home_pos":    (0.0, 0.0, 0.0),
            "sigma":       1.0,
        },
    )


@configclass
class ArmedGo2CostCfg:
    """
    Safety cost configuration for the CMDP formulation.
    Each term corresponds to an independent Lagrangian multiplier λ_i.
    """
    arm_collision: RewTerm = RewTerm(
        func=arm_collision_cost,
        weight=1.0,
        params={
            "sensor_name":     "arm_contact",
            "force_threshold": 1.0,
        },
    )
    base_proximity: RewTerm = RewTerm(
        func=base_proximity_cost,
        weight=1.0,
        params={
            "sensor_name": "lidar",
            "d_safe":      0.4,
        },
    )


# =========================================================================
# Section 4: Termination configuration
# =========================================================================

@configclass
class Go2RetrievalTerminationCfg:
    time_out    = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("base_contact"), "threshold": 1.0},
    )
    success     = DoneTerm(
        func=task_completed,
        params={"home_pos": (0.0, 0.0, 0.0), "threshold": 0.3},
    )


# =========================================================================
# Section 5: Composite action space configuration
# =========================================================================

@configclass
class ArmedGo2ActionsCfg:
    """
    Composite action space for the Go2 + Z1 mobile manipulator.
    The ActionManager concatenates these in declaration order:
        a_t = [v_x, v_y, w_z, Δq_1, ..., Δq_k]ᵀ
    """
    base: HierarchicalLocomotionActionCfg = HierarchicalLocomotionActionCfg(
        asset_name="robot",
        clip={"min": [-1.0, -0.5, -1.5], "max": [1.0, 0.5, 1.5]},
    )
    arm: RelativeJointPositionActionCfg = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint[1-6]"],  # matches the six Z1 joints confirmed above
        scale=0.1,
        use_zero_offset=True,
    )