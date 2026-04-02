"""
Stage 4: Kinematic Grasping & State Machine
Filename: collection_manager.py

Key changes from Stage 3:
- Collection condition now evaluated against the end-effector frame (p_ee)
  rather than the robot base, reducing epsilon_grasp from 0.15m to 0.05m.
- Teleportation-to-offscreen replaced by a "carried" state entry: the object
  is welded kinematically to the EE each physics step via env_manager.py.
- The observation term `target_to_ee_pos` exposes the relative displacement
  (p_target - p_ee) to the policy in place of the old base-relative vector.
"""
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg


# =========================================================================
# Section 1: Grasp Detection Logic
# =========================================================================

def retrieve_payload_logic(
    env: ManagerBasedRLEnv,
    target_cfg: SceneEntityCfg,
    ee_frame_name: str,
    threshold: float,
) -> torch.Tensor:
    """
    Vectorized grasp-detection function evaluating the EE-proximity condition.

    The collection condition is:

        d_t = ||p_ee - p_target||_2 < epsilon_grasp

    When satisfied, `has_payload` is set on the environment instance. The
    physical "welding" of the object to the arm is handled each physics step
    in `Go2RetrievalEnv._pre_physics_step`, which keeps this function a pure
    state-query with no teleportation side effects.

    Args:
        env:           The ManagerBasedRLEnv instance.
        target_cfg:    SceneEntityCfg referencing the target rigid object.
        ee_frame_name: Key of the FrameTransformerCfg sensor tracking the
                       gripper center in world coordinates.
        threshold:     epsilon_grasp in meters (~0.05 m for a 5 cm grasp zone).

    Returns:
        Float tensor of shape (num_envs, 1) acting as the payload indicator I_t.
    """
    # ObservationManager may call this before Go2RetrievalEnv.__init__ has
    # created env.has_payload. In that case, report zero payload and return.
    if not hasattr(env, "has_payload"):
        return torch.zeros(env.num_envs, 1, device=env.device, dtype=torch.float32)

    try:
        # --- 1. Acquire tensor views ---------------------------------------
        target = env.scene[target_cfg.name]
        ee_sensor = env.scene.sensors[ee_frame_name]

        # FrameTransformer: target_pos_w is (num_envs, num_target_frames, 3).
        # A single "end_effector" frame was registered in scene_cfg, so we index 0.
        p_ee = ee_sensor.data.target_pos_w[:, 0, :]   # (num_envs, 3)
        p_target = target.data.root_pos_w             # (num_envs, 3)

        # --- 2. Compute EE-to-target L2 distance ----------------------------
        dist = torch.norm(p_ee - p_target, dim=-1)    # (num_envs,)

        # --- 3. Evaluate grasp condition ------------------------------------
        # Only environments that have not yet collected the payload are eligible,
        # preventing repeated triggering once the object is already being carried.
        newly_grasped = (dist < threshold) & ~env.has_payload

        # --- 4. Persist state transition on the environment instance --------
        # This is the single authoritative write to has_payload for grasp events.
        # The kinematic attachment itself is handled in _pre_physics_step.
        if newly_grasped.any():
            env.has_payload |= newly_grasped

    except Exception as e:
        # Graceful degradation: if the EE frame sensor is misconfigured
        # (e.g., invalid target frame indices), avoid crashing the entire
        # environment and simply report no payload for this step.
        # The warning is printed once per run to avoid log spam.
        if not hasattr(env, "_payload_observation_warning_emitted"):
            print(
                "[PayloadObservationCfg] WARNING: failed to compute EE-based "
                f"payload indicator due to: {e}. Reporting zero payload."
            )
            env._payload_observation_warning_emitted = True

    # --- 5. Return the full payload indicator (including previously grasped) -
    return env.has_payload.view(-1, 1).float()


# =========================================================================
# Section 2: Relative EE Observation
# =========================================================================

def target_to_ee_pos(
    env: ManagerBasedRLEnv,
    target_cfg: SceneEntityCfg,
    ee_frame_name: str,
) -> torch.Tensor:
    """
    Computes the displacement vector from the end-effector to the target object.

    delta = p_target - p_ee

    This replaces the old base-relative target observation. Providing the
    displacement in world frame is preferable here because the arm's
    workspace is naturally expressed in world coordinates, and the policy
    already receives base orientation via `projected_gravity`.

    Args:
        env:           The ManagerBasedRLEnv instance.
        target_cfg:    SceneEntityCfg referencing the target rigid object.
        ee_frame_name: Key of the FrameTransformerCfg sensor.

    Returns:
        Float tensor of shape (num_envs, 3).
    """
    # During ObservationManager setup, or if the EE frame is misconfigured,
    # the underlying FrameTransformer may raise device-side asserts.
    # For robustness, we degrade to a zero vector in such cases so that
    # the environment can still initialize and train.
    try:
        target = env.scene[target_cfg.name]
        ee_sensor = env.scene.sensors[ee_frame_name]

        p_ee = ee_sensor.data.target_pos_w[:, 0, :]  # (num_envs, 3)
        p_target = target.data.root_pos_w            # (num_envs, 3)

        return p_target - p_ee
    except Exception as e:
        if not hasattr(env, "_target_to_ee_warning_emitted"):
            print(
                "[TargetToEEObservationCfg] WARNING: failed to compute "
                f"target-to-EE displacement due to: {e}. Returning zeros."
            )
            env._target_to_ee_warning_emitted = True
        return torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)


# =========================================================================
# Section 3: Observation Term Configurations
# =========================================================================

@configclass
class PayloadObservationCfg(ObservationTermCfg):
    """
    Maps the EE-based grasp detection logic into the MDP observation space.
    Replaces the former base-proximity PayloadObservationCfg.
    """
    func = retrieve_payload_logic
    params = {
        "target_cfg":    SceneEntityCfg("target_object"),
        "ee_frame_name": "ee_frame",
        "threshold":     0.05,  # epsilon_grasp: 5 cm grasp zone
    }


@configclass
class TargetToEEObservationCfg(ObservationTermCfg):
    """
    Exposes the world-frame displacement (p_target - p_ee) to the policy.

    This gives the high-level policy a direct arm-reaching signal and allows
    it to coordinate base locomotion with end-effector approach simultaneously.
    """
    func = target_to_ee_pos
    params = {
        "target_cfg":    SceneEntityCfg("target_object"),
        "ee_frame_name": "ee_frame",
    }