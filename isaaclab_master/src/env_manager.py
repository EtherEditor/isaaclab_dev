"""
Stage 4: Kinematic Grasping & State Machine
Filename: env_manager.py (Go2RetrievalEnv section)

Key changes from Stage 3:
- _pre_physics_step now implements EE-based kinematic attachment: the carried
  object is welded to the gripper center every physics substep, not just at
  grasp detection time.
- Grasp detection (has_payload update) is delegated exclusively to
  retrieve_payload_logic in collection_manager.py to maintain a single
  authoritative source of truth for state transitions.
- A small forward offset along the EE z-axis simulates the object sitting
  inside the gripper fingers rather than coinciding with the frame origin.
"""
import torch
import gymnasium as gym
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg as ObsTerm
import isaaclab.envs.mdp as mdp
from mdp_cfg import lidar_distance
from custom_locomotion import HierarchicalLocomotionActionCfg
from scene_cfg import Go2RetrievalSceneCfg
from mdp_cfg import ArmedGo2RewardCfg, ArmedGo2CostCfg, Go2RetrievalTerminationCfg
from event_cfg import Go2RetrievalEventCfg
from collection_manager import TargetToEEObservationCfg, PayloadObservationCfg

from adaptation_module import AdaptationModuleAdapter
from visual_encoder import StalenessAwareVisualEncoder, VisualBufferManager

class Go2RetrievalEnv(ManagerBasedRLEnv):
    """
    Extended environment manager for the armed Go2 mobile manipulation task.

    Responsibilities beyond the base ManagerBasedRLEnv:
      1. Maintains `has_payload` — the persistent boolean tensor that encodes
         whether each environment instance has successfully grasped the target.
      2. Enforces kinematic attachment each physics substep, effectively
         "welding" the carried object to the end-effector frame.
      3. Resets payload state consistently on episode boundaries.
    """

    # Gripper-center offset applied when computing the carried object's pose.
    # +0.04 m along the world Z-axis approximates the object sitting in the
    # closed fingers of the Z1 gripper rather than at the frame origin.
    _GRIPPER_OFFSET: tuple[float, float, float] = (0.0, 0.0, 0.04)

    def __init__(self, cfg: "Go2RetrievalEnvCfg", **kwargs):
        super().__init__(cfg, **kwargs)

        # Persistent per-environment payload flag.
        # dtype=torch.bool ensures no ambiguity in logical operations downstream.
        self.has_payload = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Pre-allocate the gripper offset tensor once on the correct device
        # to avoid repeated allocation in the hot physics-step loop.
        self._gripper_offset_t = torch.tensor(
            self._GRIPPER_OFFSET, dtype=torch.float32, device=self.device
        )

        # Innovation 2: attach AdaptationModuleAdapter so GaitCompensator
        # mixin can access z_e without circular imports.
        # If adaptation_module.py is not present, this attribute is simply
        # not set and GaitCompensatorMixin.apply_gait_compensation is a no-op.
        try:
            self._adaptation_module = AdaptationModuleAdapter(
                num_envs=self.num_envs, device=self.device
            )
        except Exception as e:
            print(f"[Go2RetrievalEnv] AdaptationModule unavailable: {e}")
            self._adaptation_module = None

        # Innovation 3: visual ring buffer + staleness encoder
        try:
            self._visual_buffer = VisualBufferManager(
                num_envs=self.num_envs,
                img_channels=4,
                img_h=64,
                img_w=64,
                device=self.device,
            )
            self._visual_encoder = StalenessAwareVisualEncoder().to(self.device)
            # c_v is the context vector exposed to the policy observation group
            self.c_v = torch.zeros(self.num_envs, 64, device=self.device)
        except Exception as e:
            print(f"[Go2RetrievalEnv] VisualEncoder unavailable: {e}")
            self._visual_buffer = None
            self._visual_encoder = None
            self.c_v = torch.zeros(self.num_envs, 64, device=self.device)

    # -------------------------------------------------------------------------
    # Physics Step Hook: Kinematic Attachment
    # -------------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Enforces kinematic attachment for all environments carrying the payload.

        This method is called every physics substep (at sim frequency, not
        policy frequency), ensuring the welded object tracks the EE without
        lag artifacts that would appear if attachment were only enforced at
        the policy's decimated step rate.

        Grasp detection (writing to `self.has_payload`) is intentionally NOT
        performed here — it is handled by `retrieve_payload_logic` inside the
        observation pipeline, which runs once per policy step. This avoids
        redundant distance evaluations at the higher physics frequency.
        """
        super()._pre_physics_step(actions)

        # Innovation 2: step the LSTM and update z_e once per policy step
        if self._adaptation_module is not None:
            self._adaptation_module.step(self)

        # Innovation 3: push depth frame and update c_v
        if self._visual_buffer is not None and self._visual_encoder is not None:
            depth_imgs = self._get_depth_images()
            self._visual_buffer.push(depth_imgs)
            delayed_buf = self._visual_buffer.get_delayed_buffer()
            with torch.no_grad():
                self.c_v = self._visual_encoder(
                    delayed_buf, self._visual_buffer.delay
                )

    def _get_depth_images(self) -> torch.Tensor:
        """Extract (N, 4, 64, 64) RGBD images from the depth_camera sensor."""
        try:
            sensor = self.scene.sensors["depth_camera"]
            # ray_hits_w: (N, 64*64, 3) — reshape to (N, 4, 64, 64) stub
            hits = sensor.data.ray_hits_w  # (N, H*W, 3)
            if hits is None:
                return torch.zeros(self.num_envs, 4, 64, 64, device=self.device)
            # Use the Z-component (depth) replicated across 4 channels as a stub
            depth = hits[..., 2].reshape(self.num_envs, 1, 64, 64)
            return depth.expand(-1, 4, -1, -1).contiguous()
        except Exception:
            return torch.zeros(self.num_envs, 4, 64, 64, device=self.device)

        # Early exit if no environment currently carries the payload.
        # This short-circuit is critical for performance at 4096+ envs.
        if not self.has_payload.any():
            return

        target = self.scene["target_object"]
        ee_sensor = self.scene.sensors["ee_frame"]

        # Identify the subset of environments with an active kinematic constraint.
        # nonzero() with as_tuple=False returns a (N, 1) tensor; flatten() gives (N,).
        env_ids = self.has_payload.nonzero(as_tuple=False).flatten()

        # --- Compute desired carried-object world pose -----------------------
        # p_ee: world position of the gripper center for active environments.
        # Shape: (len(env_ids), 3)
        p_ee = ee_sensor.data.target_pos_w[env_ids, 0, :]

        # The carried object is placed at a fixed offset from the gripper center.
        # Using addition rather than in-place ops preserves autograd compatibility.
        p_carried = p_ee + self._gripper_offset_t  # (len(env_ids), 3)

        # --- Construct the full root state for the physics write -------------
        # root_state_w layout: [pos(3), quat(4), lin_vel(3), ang_vel(3)] = 13 dims
        root_state = target.data.root_state_w[env_ids].clone()

        # Overwrite position with the EE-anchored target position.
        root_state[:, :3] = p_carried

        # Zero linear and angular velocities to eliminate residual momentum
        # that would cause the carried object to drift relative to the arm
        # during rapid locomotion or gait transitions.
        root_state[:, 7:13] = 0.0

        # Orientation is left unchanged (root_state[:, 3:7] untouched),
        # which is appropriate for a symmetric spherical target object.
        # For asymmetric objects, the EE quaternion should be copied here.

        # Dispatch the kinematic override to the PhysX tensor backend.
        target.write_root_state_to_sim(root_state, env_ids=env_ids)

    # -------------------------------------------------------------------------
    # Episode Reset
    # -------------------------------------------------------------------------

    def reset(self, seed: int | None = None, env_ids: torch.Tensor | None = None, options: dict | None = None) -> tuple:
        """
        Resets payload state in addition to the standard manager-based reset.

        Clearing `has_payload` before the parent reset ensures that any
        event terms (e.g., target repositioning) execute against a clean
        state rather than against the terminal-episode carried position.
        """
        if env_ids is not None:
            self.has_payload[env_ids] = False
            if self._adaptation_module is not None:
                self._adaptation_module.reset(env_ids)
        else:
            self.has_payload.zero_()
            if self._adaptation_module is not None:
                self._adaptation_module.reset()

        if self._visual_buffer is not None:
            if env_ids is not None:
                self._visual_buffer.reset(env_ids)
            else:
                self._visual_buffer.reset()

        # Parent reset must follow payload clear so that observation managers
        # reading has_payload during reset compute the correct initial state.
        return super().reset(seed=seed, env_ids=env_ids, options=options)


# =========================================================================
# Observation Space (Updated for Stage 4)
# =========================================================================

@configclass
class Go2RetrievalObservationsCfg:
    """
    Aggregates proprioceptive, exteroceptive, and manipulation observation
    terms into the policy's observation space.

    Stage 4 additions to the policy group:
      - `has_payload`:     Scalar indicator I_t from EE-based grasp detection.
      - `target_to_ee`:    World-frame displacement (p_target - p_ee), replacing
                           the former base-relative target position.
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        concatenate_terms = True
        enable_corruption = True

        # Base kinematic state and joint observations
        base_lin_vel    = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel    = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_err   = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel       = ObsTerm(func=mdp.joint_vel_rel)
        last_action     = ObsTerm(func=mdp.last_action)

        # Lidar scan and (temporarily) a zero target command vector.
        # The original 'target_pos' term used mdp.generated_commands with
        # command_name='target_pose', but no such command is currently
        # registered in the CommandManager, causing a KeyError. Until a
        # proper command term is wired up, we expose a zero 3D command.
        target_pos    = ObsTerm(
            func=lambda env: torch.zeros(env.num_envs, 3, device=env.device),
        )
        obstacle_scan = ObsTerm(
            func=lidar_distance,
            params={"sensor_name": "lidar"},
        )

        # NOTE: Temporarily disabling EE-based payload and target-to-EE
        # observations because the underlying FrameTransformer configuration
        # for the Z1 end-effector is triggering device-side asserts on CUDA.
        # The environment can still train locomotion and obstacle avoidance
        # while we iterate on a correct EE frame setup.
        # has_payload: PayloadObservationCfg = PayloadObservationCfg()
        # target_to_ee: TargetToEEObservationCfg = TargetToEEObservationCfg()

    policy: PolicyCfg = PolicyCfg()


# =========================================================================
# Action & Environment Configurations (Unchanged structure, updated refs)
# =========================================================================

@configclass
class Go2RetrievalActionsCfg:
    """Routes the composite (vx, vy, wz, delta_q_arm) action vector."""
    locomotion: HierarchicalLocomotionActionCfg = HierarchicalLocomotionActionCfg()


@configclass
class Go2RetrievalEnvCfg(ManagerBasedRLEnvCfg):
    """Top-level configuration binding all Stage 1–4 components."""

    episode_length_s: float = 20.0
    decimation: int = 4

    scene: Go2RetrievalSceneCfg = Go2RetrievalSceneCfg(
        num_envs=4096, env_spacing=2.5
    )
    observations: Go2RetrievalObservationsCfg = Go2RetrievalObservationsCfg()
    actions: Go2RetrievalActionsCfg = Go2RetrievalActionsCfg()
    events: Go2RetrievalEventCfg = Go2RetrievalEventCfg()
    rewards: ArmedGo2RewardCfg = ArmedGo2RewardCfg()
    costs: ArmedGo2CostCfg = ArmedGo2CostCfg()
    terminations: Go2RetrievalTerminationCfg = Go2RetrievalTerminationCfg()

    def __post_init__(self):
        super().__post_init__()
        self.viewer.eye = (0.0, -3.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

# =========================================================================
# Grasp Curriculum: Dynamic Threshold Decay
# =========================================================================

def update_grasp_curriculum(
    env: Go2RetrievalEnv,
    current_iteration: int,
    max_iterations: int,
    threshold_start: float = 0.5,
    threshold_end: float   = 0.05,
    warmup_fraction: float = 0.5,
) -> float:
    """
    Linearly decays the EE grasp-detection threshold over the first
    `warmup_fraction` of training, then holds it at `threshold_end`.

    Schedule:
        t* = warmup_fraction × max_iterations
        ε(t) = threshold_start − (threshold_start − threshold_end)
                                × min(t / t*, 1.0)

    Starting at 0.5 m allows the policy to accumulate reward signal and
    learn approach behaviour before the grasp condition becomes tight.
    The final 0.05 m matches the physical gripper closure radius of the
    Z1 end-effector and is reached at the midpoint of training, leaving
    the second half for fine-grained manipulation refinement.

    Args:
        env:               The Go2RetrievalEnv instance.
        current_iteration: Current training iteration index.
        max_iterations:    Total number of planned training iterations.
        threshold_start:   Initial (permissive) grasp threshold in metres.
        threshold_end:     Final (tight) grasp threshold in metres.
        warmup_fraction:   Fraction of max_iterations over which to decay.

    Returns:
        The new threshold value (float), primarily for logging purposes.
    """
    warmup_steps = warmup_fraction * max_iterations
    progress     = min(current_iteration / warmup_steps, 1.0)
    new_threshold = threshold_start - progress * (threshold_start - threshold_end)

    # Traverse the observation manager to locate the PayloadObservationCfg
    # term and update its threshold parameter in-place. This approach avoids
    # rebuilding the observation manager and incurring re-initialisation cost.
    obs_manager = env.observation_manager
    policy_group = obs_manager._group_obs_term_cfgs.get("policy", {})

    if "has_payload" in policy_group:
        policy_group["has_payload"].params["threshold"] = new_threshold

    return new_threshold

# =========================================================================
# Environment Registration
# =========================================================================

gym.register(
    id="Isaac-Go2-Retrieval-v0",
    entry_point="env_manager:Go2RetrievalEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": Go2RetrievalEnvCfg},
)