"""
RMA Adaptation Module: LSTM-based arm inertia estimator.
Filename: adaptation_module.py

Trained with an L2 distillation loss against the teacher policy's privileged
state (arm_inertia_tensor + payload_com_offset). The resulting 16-dim
embedding z_e is appended to the student policy's observation vector.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# LSTM adaptation module
# =============================================================================

class AdaptationModule(nn.Module):
    """
    2-layer LSTM that maps a rolling history of proprioceptive signals to a
    16-dim extrinsics embedding z_e encoding inferred arm inertia.

    Input per step  : [12 joint torques + 4 contact forces] = 16 floats
    History length  : T = 10 steps
    Output          : z_e (16,)
    """

    INPUT_DIM: int  = 16   # 12 torques + 4 contact forces
    HIDDEN_DIM: int = 64
    NUM_LAYERS: int = 2
    EMBED_DIM: int  = 16
    HISTORY_LEN: int = 10

    def __init__(self) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=self.INPUT_DIM,
            hidden_size=self.HIDDEN_DIM,
            num_layers=self.NUM_LAYERS,
            batch_first=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(self.HIDDEN_DIM, self.EMBED_DIM),
            nn.Tanh(),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: (B, T, INPUT_DIM) rolling proprioceptive history.
        Returns:
            z_e: (B, EMBED_DIM)
        """
        lstm_out, _ = self.lstm(history)      # (B, T, HIDDEN_DIM)
        z_e = self.proj(lstm_out[:, -1, :])  # use last timestep
        return z_e

    def distillation_loss(
        self,
        z_e: torch.Tensor,
        privileged_state: torch.Tensor,
        weight: float = 0.1,
    ) -> torch.Tensor:
        """
        L2 regression loss against the teacher's privileged state vector.

        L = weight * ||z_e - W_priv @ privileged_state||^2

        We project privileged_state (9-dim: 6 inertia + 3 CoM) to EMBED_DIM
        via a lazy-initialized linear layer stored as a buffer.
        """
        if not hasattr(self, "_priv_proj"):
            priv_dim = privileged_state.shape[-1]
            self._priv_proj = nn.Linear(priv_dim, self.EMBED_DIM, bias=False).to(z_e.device)

        target = self._priv_proj(privileged_state.detach())
        return weight * F.mse_loss(z_e, target)


import torch.nn.functional as F


# =============================================================================
# Environment-facing adapter: maintains the history buffer and calls the LSTM
# =============================================================================

class AdaptationModuleAdapter:
    """
    Stateful wrapper around AdaptationModule that maintains the rolling
    history buffer and exposes `z_e` as a per-environment GPU tensor.

    Lifecycle
    ---------
    - Instantiated in Go2RetrievalEnv.__init__().
    - step() called in _pre_physics_step() every policy step.
    - reset() called on episode reset.
    """

    def __init__(self, num_envs: int, device: str | torch.device) -> None:
        self.num_envs = num_envs
        self.device   = device

        self._module = AdaptationModule().to(device)

        # Rolling history buffer: (N, T, INPUT_DIM)
        self._history = torch.zeros(
            num_envs,
            AdaptationModule.HISTORY_LEN,
            AdaptationModule.INPUT_DIM,
            device=device,
        )

        # Latest embedding output: (N, EMBED_DIM)
        self.z_e = torch.zeros(num_envs, AdaptationModule.EMBED_DIM, device=device)

    # -------------------------------------------------------------------------

    def step(self, env: "ManagerBasedRLEnv") -> None:
        """
        Collect current proprioceptive signals, update history, and run LSTM.

        Must be called once per policy step (not substep).
        """
        torques  = self._get_joint_torques(env)   # (N, 12)
        contacts = self._get_contact_forces(env)  # (N, 4)

        new_obs = torch.cat([torques, contacts], dim=-1)  # (N, 16)

        # Shift history left and append new observation
        self._history = torch.roll(self._history, shifts=-1, dims=1)
        self._history[:, -1, :] = new_obs

        # Run the LSTM (no grad needed during env interaction)
        with torch.no_grad():
            self.z_e = self._module(self._history)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None:
            self._history[env_ids] = 0.0
            self.z_e[env_ids] = 0.0
        else:
            self._history.zero_()
            self.z_e.zero_()

    def parameters(self):
        return self._module.parameters()

    def state_dict(self):
        return self._module.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self._module.load_state_dict(sd)

    # -------------------------------------------------------------------------
    # Proprioceptive signal extraction
    # -------------------------------------------------------------------------

    def _get_joint_torques(self, env: "ManagerBasedRLEnv") -> torch.Tensor:
        """Extract joint torques from the articulation (12-DOF leg joints)."""
        try:
            robot = env.scene["robot"]
            # applied_torque is (N, total_joints); take the first 12 (legs)
            torques = robot.data.applied_torque[:, :12]
        except Exception:
            torques = torch.zeros(self.num_envs, 12, device=self.device)
        return torques

    def _get_contact_forces(self, env: "ManagerBasedRLEnv") -> torch.Tensor:
        """
        Extract per-foot contact force magnitudes (4 feet).
        Uses the ContactSensor attached to the foot links.
        """
        try:
            sensor = env.scene.sensors["base_contact"]
            forces = sensor.data.net_forces_w           # (N, B, 3)
            # Take the first 4 bodies as foot contacts; sum magnitude
            foot_f = torch.norm(forces[:, :4, :], dim=-1)  # (N, 4)
        except Exception:
            foot_f = torch.zeros(self.num_envs, 4, device=self.device)
        return foot_f