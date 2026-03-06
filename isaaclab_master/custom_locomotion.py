"""
Phase 5: Hierarchical Action Wrapper (Corrected)
Filename: custom_locomotion.py
"""
import os
import torch
from typing import TYPE_CHECKING
from isaaclab.utils import configclass
from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Use string annotations ("...") for the configuration class to defer evaluation
class HierarchicalLocomotionAction(ActionTerm):
    """
    Highly optimized action term scaling normalized RL commands and executing 
    zero-copy inferences against a pre-trained locomotion policy.
    """
    cfg: "HierarchicalLocomotionActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "HierarchicalLocomotionActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        
        # Resident GPU Scaling Tensors
        self._velocity_scale = torch.tensor(self.cfg.velocity_scale, device=self.device)
        self._joint_action_scale = self.cfg.joint_action_scale
        self._default_joint_pos = torch.tensor(self.cfg.default_joint_pos, device=self.device)
        
        # Load and optimize LibTorch policy when available.
        # If the file is missing, fall back to a deterministic
        # direct joint-control heuristic so that the environment
        # can still initialize and train end-to-end.
        self._low_level_policy = None
        if os.path.exists(self.cfg.policy_path):
            try:
                self._low_level_policy = torch.jit.load(self.cfg.policy_path).to(self.device)
                self._low_level_policy.eval()
                self._low_level_policy = torch.jit.optimize_for_inference(self._low_level_policy)
            except Exception as e:
                raise RuntimeError(f"Failed to load low-level policy from {self.cfg.policy_path}: {e}")
        else:
            print(
                f"[HierarchicalLocomotionAction] WARNING: "
                f"low-level policy file not found at {self.cfg.policy_path}. "
                "Falling back to direct joint-space control."
            )

        # Strictly Pre-allocated Buffers (48-dim state)
        self._low_level_obs = torch.zeros((self.num_envs, 48), device=self.device)
        self._last_raw_actions = torch.zeros((self.num_envs, 12), device=self.device)

    # ... keep properties and methods (action_dim, apply_actions, reset) exactly as they are ...
    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions = actions
        self._processed_actions = actions * self._velocity_scale

    def apply_actions(self) -> None:
        """
        Executes zero-copy state construction and applies the residual joint mapping:
        q_des = q_default + c_scale * pi_low(obs_low)
        """
        if self._low_level_policy is not None:
            # 1. Zero-copy state assembly via in-place slicing to prevent memory reallocation
            self._low_level_obs[:, 0:3] = self._asset.data.root_lin_vel_b
            self._low_level_obs[:, 3:6] = self._asset.data.root_ang_vel_b
            self._low_level_obs[:, 6:9] = self._asset.data.projected_gravity_b
            self._low_level_obs[:, 9:12] = self._processed_actions
            
            # Provide joint position error rather than absolute joint positions
            self._low_level_obs[:, 12:24] = self._asset.data.joint_pos - self._default_joint_pos
            self._low_level_obs[:, 24:36] = self._asset.data.joint_vel
            self._low_level_obs[:, 36:48] = self._last_raw_actions

            # 2. Query Low-Level Policy
            with torch.no_grad():
                raw_q_des = self._low_level_policy(self._low_level_obs)
        else:
            # Deterministic fallback: map the 3D velocity command directly
            # into 12 joint residuals using a simple heuristic. This keeps
            # the action interface unchanged while allowing end-to-end
            # learning without a pre-trained low-level controller.
            a = self._processed_actions  # (num_envs, 3): [v_x, v_y, w_z]
            raw_q_des = torch.zeros(self.num_envs, 12, device=self.device)

            # Hip roll (lateral motion via v_y)
            raw_q_des[:, 0] = a[:, 1]      # FL hip roll
            raw_q_des[:, 3] = -a[:, 1]     # FR hip roll
            raw_q_des[:, 6] = a[:, 1]      # RL hip roll
            raw_q_des[:, 9] = -a[:, 1]     # RR hip roll

            # Hip pitch (forward motion via v_x)
            raw_q_des[:, 1] = a[:, 0]      # FL hip pitch
            raw_q_des[:, 4] = a[:, 0]      # FR hip pitch
            raw_q_des[:, 7] = a[:, 0]      # RL hip pitch
            raw_q_des[:,10] = a[:, 0]      # RR hip pitch

            # Knee flexion increases with forward speed magnitude
            knee_flex = -torch.abs(a[:, 0])
            raw_q_des[:, 2] = knee_flex    # FL knee
            raw_q_des[:, 5] = knee_flex    # FR knee
            raw_q_des[:, 8] = knee_flex    # RL knee
            raw_q_des[:,11] = knee_flex    # RR knee

        # 3. Affine transform: Compute absolute joint targets from normalized residuals
        q_des = self._default_joint_pos + self._joint_action_scale * raw_q_des
            
        # 4. Dispatch to Physics Engine Interface
        self._asset.set_joint_position_target(q_des)
        
        # Overwrite in-place to preserve memory address
        self._last_raw_actions[:] = raw_q_des

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None:
            self._last_raw_actions[env_ids] = 0.0
        else:
            self._last_raw_actions.zero_()


@configclass
class HierarchicalLocomotionActionCfg(ActionTermCfg):
    """Configuration for the Hierarchical Locomotion Action Term."""
    class_type: type = HierarchicalLocomotionAction
    asset_name: str = "robot"
    # Dynamically resolve absolute path relative to this Python file
    policy_path: str = os.path.join(os.path.dirname(__file__), "models", "go2_low_level_locomotion.pt")
    
    # RL Action Space Bounding
    clip: dict = {"min": [-1.0, -0.5, -1.5], "max": [1.0, 0.5, 1.5]}
    
    velocity_scale: tuple[float, float, float] = (1.0, 0.5, 1.5)
    joint_action_scale: float = 0.25
    default_joint_pos: tuple[float, ...] = (
        0.1, 0.8, -1.5,   # FL
        -0.1, 0.8, -1.5,  # FR
        0.1, 1.0, -1.5,   # RL
        -0.1, 1.0, -1.5   # RR
    )


# Bind the dynamic class type mapping
HierarchicalLocomotionActionCfg.class_type = HierarchicalLocomotionAction