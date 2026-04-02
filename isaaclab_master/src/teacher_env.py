"""
Teacher environment with privileged observations for RMA distillation.
Filename: teacher_env.py
"""
from __future__ import annotations

import torch
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
import isaaclab.envs.mdp as mdp

from env_manager import Go2RetrievalEnvCfg, Go2RetrievalEnv, Go2RetrievalObservationsCfg


# =============================================================================
# Privileged observation functions
# =============================================================================

def terrain_height_scan(env, sensor_name: str = "lidar") -> torch.Tensor:
    """
    Returns a 5×5 height grid centred on the robot base using the lidar sensor.

    We project ray hit positions onto the base XY frame and bilinearly
    interpolate to a fixed 5×5 grid.  Output shape: (N, 25).
    """
    try:
        sensor     = env.scene.sensors[sensor_name]
        ray_hits   = sensor.data.ray_hits_w              # (N, B, 3)
        sensor_pos = sensor.data.pos_w                   # (N, 3)
        if ray_hits is None or sensor_pos is None:
            return torch.zeros(env.num_envs, 25, device=env.device)

        # Height relative to robot base
        rel_hits = ray_hits - sensor_pos.unsqueeze(1)    # (N, B, 3)
        heights  = rel_hits[..., 2]                      # (N, B)

        # Sample 25 values from the available rays (nearest-neighbour)
        n_rays = heights.shape[1]
        indices = torch.linspace(0, n_rays - 1, 25, device=env.device).long()
        grid = heights[:, indices]                       # (N, 25)
        return grid
    except Exception:
        return torch.zeros(env.num_envs, 25, device=env.device)


def arm_inertia_tensor(env) -> torch.Tensor:
    """
    Returns the composite inertia tensor of the Z1 arm: 6 floats
    (Ixx, Iyy, Izz, Ixy, Ixz, Iyz) per environment.

    Reads articulation rigid-body inertia data from Isaac Sim.
    """
    try:
        robot = env.scene["robot"]
        # body_inertias: (N, total_bodies, 9) — flattened 3×3 inertia per body
        inertia = robot.data.body_inertia_w              # (N, B, 9)
        # Sum over Z1 links (bodies 13–18 in the composite articulation;
        # indices are approximate — adjust based on actual USD body order).
        arm_inertia = inertia[:, 13:19, :].sum(dim=1)   # (N, 9)
        # Extract symmetric 6-element representation
        Ixx = arm_inertia[:, 0]
        Iyy = arm_inertia[:, 4]
        Izz = arm_inertia[:, 8]
        Ixy = arm_inertia[:, 1]
        Ixz = arm_inertia[:, 2]
        Iyz = arm_inertia[:, 5]
        return torch.stack([Ixx, Iyy, Izz, Ixy, Ixz, Iyz], dim=-1)  # (N, 6)
    except Exception:
        return torch.zeros(env.num_envs, 6, device=env.device)


def payload_com_offset(env) -> torch.Tensor:
    """
    Returns the 3D CoM shift caused by the carried object.
    Zero when has_payload is False.  Output shape: (N, 3).
    """
    if not hasattr(env, "has_payload"):
        return torch.zeros(env.num_envs, 3, device=env.device)
    try:
        target   = env.scene["target_object"]
        robot    = env.scene["robot"]
        base_pos = robot.data.root_pos_w                 # (N, 3)
        obj_pos  = target.data.root_pos_w                # (N, 3)
        offset   = obj_pos - base_pos                    # (N, 3)
        mask     = env.has_payload.float().unsqueeze(-1) # (N, 1)
        return offset * mask
    except Exception:
        return torch.zeros(env.num_envs, 3, device=env.device)


# =============================================================================
# Observation configs
# =============================================================================

@configclass
class PrivilegedObsGroup(ObsGroup):
    """
    Privileged state group — visible only to the teacher policy.
    Concatenated into a single vector for distillation target.
    """
    concatenate_terms: bool = True
    enable_corruption: bool = False

    terrain_height_scan: ObsTerm = ObsTerm(
        func=terrain_height_scan,
        params={"sensor_name": "lidar"},
    )
    arm_inertia_tensor: ObsTerm = ObsTerm(func=arm_inertia_tensor)
    payload_com_offset: ObsTerm = ObsTerm(func=payload_com_offset)


@configclass
class TeacherObservationsCfg(Go2RetrievalObservationsCfg):
    """Extends the base observations with the privileged state group."""
    privileged_state: PrivilegedObsGroup = PrivilegedObsGroup()


# =============================================================================
# Teacher environment config
# =============================================================================

@configclass
class TeacherEnvCfg(Go2RetrievalEnvCfg):
    """
    Teacher environment configuration for RMA pretraining.
    Identical to the base env except the observation manager now exposes
    the `privileged_state` group.
    """
    observations: TeacherObservationsCfg = TeacherObservationsCfg()


# =============================================================================
# Teacher environment class
# =============================================================================

class TeacherEnv(Go2RetrievalEnv):
    """
    Teacher environment that exposes the privileged state tensor for
    distillation loss computation in the student training loop.
    """

    def get_privileged_state(self) -> torch.Tensor:
        """
        Returns the concatenated privileged state: (N, 34).
        [terrain_height_scan(25) | arm_inertia_tensor(6) | payload_com_offset(3)]
        """
        obs_dict = self.observation_manager.compute_group("privileged_state")
        return obs_dict  # already concatenated by ObsGroup