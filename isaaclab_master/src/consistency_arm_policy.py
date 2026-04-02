"""
Consistency Policy diffusion head for the Z1 arm.
Filename: consistency_arm_policy.py

Replaces RelativeJointPositionAction with a 1-step Consistency Policy (CP-CD)
whose inference is steered by an SDF collision-energy gradient at every step.
No backward pass through the network occurs during inference.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# =============================================================================
# U-Net building blocks (≤3 down-blocks, hidden=128)
# =============================================================================

class ResBlock1D(nn.Module):
    """1-D residual block used in the U-Net backbone."""

    def __init__(self, channels: int, sigma_embed_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        # project sigma embedding into the channel dimension
        self.sigma_proj = nn.Linear(sigma_embed_dim, channels)

    def forward(self, x: torch.Tensor, sigma_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)   sigma_emb: (B, sigma_embed_dim)
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.sigma_proj(sigma_emb).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class ConsistencyUNet(nn.Module):
    """
    Lightweight U-Net denoiser for the Consistency Policy.

    Maps (obs, noisy_action, sigma) → denoised_action.

    Architecture
    ------------
    - obs and noisy_action are projected to hidden_dim=128.
    - 3 down-blocks (Conv1d → ResBlock) with spatial length L=1 (the 6-DOF
      action vector is treated as a sequence of length 1 with C channels).
    - Symmetric skip-connected up-path.
    - sigma embedded via sinusoidal + MLP following Karras 2022.
    """

    SIGMA_EMBED_DIM: int = 64
    HIDDEN: int = 128

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim

        # Sigma embedding: sinusoidal → MLP
        self.sigma_mlp = nn.Sequential(
            nn.Linear(self.SIGMA_EMBED_DIM, self.SIGMA_EMBED_DIM * 2),
            nn.SiLU(),
            nn.Linear(self.SIGMA_EMBED_DIM * 2, self.SIGMA_EMBED_DIM),
        )

        # Input projection (concat obs + noisy_action → hidden)
        self.in_proj = nn.Linear(obs_dim + action_dim, self.HIDDEN)

        # Down-blocks
        self.down1 = ResBlock1D(self.HIDDEN, self.SIGMA_EMBED_DIM)
        self.down2 = ResBlock1D(self.HIDDEN, self.SIGMA_EMBED_DIM)
        self.down3 = ResBlock1D(self.HIDDEN, self.SIGMA_EMBED_DIM)

        # Mid-block
        self.mid = ResBlock1D(self.HIDDEN, self.SIGMA_EMBED_DIM)

        # Up-blocks (with skip connections → 2×HIDDEN input)
        self.up_proj3 = nn.Conv1d(self.HIDDEN * 2, self.HIDDEN, 1)
        self.up3 = ResBlock1D(self.HIDDEN, self.SIGMA_EMBED_DIM)
        self.up_proj2 = nn.Conv1d(self.HIDDEN * 2, self.HIDDEN, 1)
        self.up2 = ResBlock1D(self.HIDDEN, self.SIGMA_EMBED_DIM)
        self.up_proj1 = nn.Conv1d(self.HIDDEN * 2, self.HIDDEN, 1)
        self.up1 = ResBlock1D(self.HIDDEN, self.SIGMA_EMBED_DIM)

        # Output projection back to action_dim
        self.out_norm = nn.GroupNorm(8, self.HIDDEN)
        self.out_proj = nn.Linear(self.HIDDEN, action_dim)

    # -------------------------------------------------------------------------
    # Sinusoidal sigma embedding (Karras 2022, eq. 7)
    # -------------------------------------------------------------------------
    def _sigma_embed(self, sigma: torch.Tensor) -> torch.Tensor:
        """sigma: (B,) → (B, SIGMA_EMBED_DIM)"""
        half = self.SIGMA_EMBED_DIM // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=sigma.device) / (half - 1)
        )
        args = sigma.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.sigma_mlp(emb)

    def forward(
        self,
        obs: torch.Tensor,           # (B, obs_dim)
        noisy_action: torch.Tensor,  # (B, action_dim)
        sigma: torch.Tensor,         # (B,)
    ) -> torch.Tensor:
        """Returns denoised action of shape (B, action_dim)."""
        sigma_emb = self._sigma_embed(sigma)              # (B, SIGMA_EMBED_DIM)

        x = self.in_proj(torch.cat([obs, noisy_action], dim=-1))  # (B, HIDDEN)
        x = x.unsqueeze(-1)                              # (B, HIDDEN, 1)

        # Down-path
        h1 = self.down1(x, sigma_emb)
        h2 = self.down2(h1, sigma_emb)
        h3 = self.down3(h2, sigma_emb)

        # Mid
        m = self.mid(h3, sigma_emb)

        # Up-path with skip connections
        u3 = self.up3(self.up_proj3(torch.cat([m, h3], dim=1)), sigma_emb)
        u2 = self.up2(self.up_proj2(torch.cat([u3, h2], dim=1)), sigma_emb)
        u1 = self.up1(self.up_proj1(torch.cat([u2, h1], dim=1)), sigma_emb)

        # Output
        out = F.silu(self.out_norm(u1)).squeeze(-1)      # (B, HIDDEN)
        return self.out_proj(out)                        # (B, action_dim)


# =============================================================================
# Consistency Training (CT) loss  (Karras 2022 / Song 2023)
# =============================================================================

def consistency_training_loss(
    model: ConsistencyUNet,
    obs: torch.Tensor,
    actions: torch.Tensor,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    n_steps: int = 150,
) -> torch.Tensor:
    """
    Consistency Training (CT) loss for one minibatch.

    L_CT = E[ w(sigma_n) * ||f(x_n, sigma_n) - f(x_{n-1}, sigma_{n-1})||^2 ]

    where x_k = actions + sigma_k * eps,  eps ~ N(0,I).
    The stop-gradient target uses the EMA model (approximated here by
    detaching — a production implementation should maintain an EMA copy).
    """
    B = obs.shape[0]
    device = obs.device

    # Sample random adjacent step indices for each batch element
    n = torch.randint(1, n_steps, (B,), device=device)

    # Compute sigma schedule (Karras eq. 2)
    def sigma_schedule(k: torch.Tensor) -> torch.Tensor:
        return (
            sigma_min ** (1.0 / rho)
            + (k / (n_steps - 1)) * (sigma_max ** (1.0 / rho) - sigma_min ** (1.0 / rho))
        ) ** rho

    sigma_n = sigma_schedule(n.float())          # (B,)
    sigma_nm1 = sigma_schedule((n - 1).float())  # (B,)

    eps = torch.randn_like(actions)

    x_n   = actions + sigma_n.unsqueeze(-1) * eps
    x_nm1 = actions + sigma_nm1.unsqueeze(-1) * eps

    # Loss weighting (Karras eq. 3)
    w = 1.0 / (sigma_n - sigma_nm1 + 1e-5)

    # Online model prediction
    pred_n = model(obs, x_n, sigma_n)

    # Stop-gradient target (EMA approximated by detach)
    with torch.no_grad():
        target_nm1 = model(obs, x_nm1, sigma_nm1)

    loss = (w.unsqueeze(-1) * (pred_n - target_nm1) ** 2).mean()
    return loss


# =============================================================================
# ConsistencyArmPolicy — ActionTerm
# =============================================================================

class ConsistencyArmPolicy(ActionTerm):
    """
    1-step Consistency Policy arm action term with SDF energy guidance.

    Action pipeline per step
    ------------------------
    1. Sample z ~ N(0, I_6) (sigma_max level noise).
    2. Single denoising step: a_raw = f(obs, z, sigma_max).
    3. SDF guidance correction: a_guided = a_raw - alpha * grad_a E_SDF(FK(a_raw)).
       Gradient computed via finite differences (no network backward pass).
    4. Clip to joint limits and dispatch to articulation.
    """

    cfg: "ConsistencyArmActionCfg"

    SIGMA_MAX: float = 80.0
    SIGMA_MIN: float = 0.002

    def __init__(self, cfg: "ConsistencyArmActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)

        self._action_dim: int = cfg.action_dim
        obs_dim: int = cfg.obs_dim

        # Build denoiser network and move to env device
        self._model = ConsistencyUNet(obs_dim, self._action_dim).to(self.device)

        # Import SDF module (may be None if no SDF has been built yet)
        self._sdf_guidance = None
        try:
            from sdf_guidance import SDFGuidance
            self._sdf_guidance = SDFGuidance(
                alpha=cfg.sdf_alpha,
                d_safe=cfg.sdf_d_safe,
                fd_eps=cfg.sdf_fd_eps,
                device=self.device,
            )
        except ImportError:
            print("[ConsistencyArmPolicy] sdf_guidance not found; SDF guidance disabled.")

        # Pre-allocate action buffers
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)

        # sigma_max tensor for single denoising step
        self._sigma_max_t = torch.full((self.num_envs,), self.SIGMA_MAX, device=self.device)

    # -------------------------------------------------------------------------
    # ActionTerm interface
    # -------------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Store the high-level actions (used as conditioning obs, not arm cmd)."""
        self._raw_actions = actions

    def apply_actions(self) -> None:
        """Run 1-step diffusion + optional SDF guidance and dispatch joint targets."""
        # 1. Assemble observation vector from robot state
        obs = self._build_obs()                          # (N, obs_dim)

        # 2. Sample noise at sigma_max
        z = torch.randn(self.num_envs, self._action_dim, device=self.device) * self.SIGMA_MAX

        # 3. Single denoising step (no grad needed for the network)
        with torch.no_grad():
            a_raw = self._model(obs, z, self._sigma_max_t)   # (N, action_dim)

        # 4. SDF guidance (finite-difference gradient — no network backward pass)
        if self._sdf_guidance is not None and self._sdf_guidance.is_ready():
            a_guided = self._sdf_guidance.apply_guidance(a_raw)
        else:
            a_guided = a_raw

        # 5. Scale and clip
        a_clipped = torch.clamp(
            a_guided * self.cfg.scale,
            self.cfg.joint_limits[0],
            self.cfg.joint_limits[1],
        )
        self._processed_actions = a_clipped

        # 6. Dispatch relative joint position target
        asset = self.scene[self.cfg.asset_name]
        current_q = asset.data.joint_pos[:, self._joint_ids]
        target_q = current_q + a_clipped
        asset.set_joint_position_target(target_q, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
        else:
            self._raw_actions.zero_()
            self._processed_actions.zero_()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _build_obs(self) -> torch.Tensor:
        """Construct the conditioning observation for the denoiser."""
        asset = self.scene[self.cfg.asset_name]
        # Use proprioception: base vel (6) + gravity (3) + joint_pos (N_j) + joint_vel (N_j)
        base_lin = asset.data.root_lin_vel_b           # (N, 3)
        base_ang = asset.data.root_ang_vel_b           # (N, 3)
        gravity  = asset.data.projected_gravity_b      # (N, 3)
        jpos     = asset.data.joint_pos                # (N, total_joints)
        jvel     = asset.data.joint_vel                # (N, total_joints)

        # Concatenate; will be truncated / padded to cfg.obs_dim via projection
        obs = torch.cat([base_lin, base_ang, gravity, jpos, jvel], dim=-1)

        # Clip to expected obs_dim (pad with zeros if articulation has fewer joints)
        obs_dim = self.cfg.obs_dim
        if obs.shape[-1] < obs_dim:
            pad = torch.zeros(self.num_envs, obs_dim - obs.shape[-1], device=self.device)
            obs = torch.cat([obs, pad], dim=-1)
        else:
            obs = obs[:, :obs_dim]
        return obs

    @property
    def _joint_ids(self) -> list[int]:
        """Cached joint indices for Z1 arm joints."""
        if not hasattr(self, "_cached_joint_ids"):
            asset = self.scene[self.cfg.asset_name]
            self._cached_joint_ids = [
                asset.find_joints(expr)[0]
                for expr in self.cfg.joint_names
            ]
            # flatten if find_joints returns lists
            flat = []
            for item in self._cached_joint_ids:
                if isinstance(item, (list, tuple)):
                    flat.extend(item)
                else:
                    flat.append(item)
            self._cached_joint_ids = flat
        return self._cached_joint_ids


# =============================================================================
# Configuration
# =============================================================================

@configclass
class ConsistencyArmActionCfg(ActionTermCfg):
    """Configuration for the Consistency Policy arm action term."""

    class_type: type = ConsistencyArmPolicy

    asset_name: str = "robot"
    joint_names: list[str] = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    action_dim: int = 6
    obs_dim: int = 48          # must match the denoiser input dimension
    scale: float = 0.1
    joint_limits: tuple[float, float] = (-3.14, 3.14)

    # SDF guidance hyperparameters
    sdf_alpha: float = 0.05    # step size for the SDF correction
    sdf_d_safe: float = 0.10   # safety margin in metres
    sdf_fd_eps: float = 1e-3   # finite-difference epsilon