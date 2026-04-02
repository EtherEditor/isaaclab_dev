"""
Stage 5: PPO-Lagrangian Algorithm & Training Entry Point
Filename: train.py

Implements a Constrained MDP (CMDP) training loop via a custom
ConstrainedOnPolicyRunner that extends RSL-RL's OnPolicyRunner with:
  - A dual critic architecture (V_r and V_c).
  - Per-constraint learnable Lagrangian multipliers updated via dual ascent.
  - A modified PPO objective: L_total = L_policy + λ·L_cost + c1·L_vf - c2·H
"""

import argparse
import os
import time
import logging
from datetime import datetime
from dataclasses import dataclass, field

# =========================================================================
# 1. Isaac Sim Bootstrap (must precede all other imports)
# =========================================================================

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Train Armed Go2 Retrieval Policy with PPO-Lagrangian."
)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument(
    "--resume_path", type=str, default=None,
    help="Path to a checkpoint directory to resume training from."
)
# --- Stage 0: Experiment group configuration ---
parser.add_argument(
    "--track", type=str, default="sampling",
    choices=["sampling", "sim2real", "vla_grasp"],
    help="Innovation track to activate (default: sampling).",
)
parser.add_argument(
    "--group", type=str, default="B0",
    choices=["B0", "E1", "E2", "E3", "E4"],
    help="Experiment group label (default: B0).",
)
parser.add_argument(
    "--dry-run", action="store_true", default=False,
    help="Smoke-test: instantiate env, step 10 times with random actions, then exit.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =========================================================================
# 2. Post-launch imports
# =========================================================================

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)

import env_manager
from env_manager import Go2RetrievalEnvCfg
from mdp_cfg import arm_collision_cost, base_proximity_cost
import experiment_cfg


# Performance tuning for Ampere/Hopper-class GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Module-level logger — avoids print() calls in hot paths
log = logging.getLogger("go2_retrieval")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# =========================================================================
# 4. Dual-Critic Actor-Critic Network
# =========================================================================

# =========================================================================
# RunningMeanStd: GPU-resident online normalizer (Welford's algorithm)
# =========================================================================

class RunningMeanStd(nn.Module):
    """
    Tracks the running mean and variance of a tensor stream using
    Welford's online algorithm.

    All statistics are maintained on the target device to avoid
    host-device transfers in the hot rollout-collection path.

    Args:
        shape:   Shape of a single observation vector, e.g. (obs_dim,).
        epsilon: Small constant added to the variance denominator to
                 prevent division-by-zero on the first update.
        clip:    Symmetric clamp applied to the normalised output.
                 A value of 5.0 is standard in PPO implementations and
                 prevents extreme inputs from destabilising early training.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        epsilon: float = 1e-8,
        clip: float = 5.0,
        device: str = "cuda",
    ):
        super().__init__()
        self.epsilon = epsilon
        self.clip    = clip

        # Registered as buffers so they are included in state_dict()
        # and moved correctly by .to(device).
        self.register_buffer("mean",  torch.zeros(shape, device=device))
        self.register_buffer("var",   torch.ones(shape,  device=device))
        self.register_buffer("count", torch.tensor(epsilon, device=device))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """
        Ingests a batch of observations and updates running statistics.

        Args:
            x: Tensor of shape (batch_size, *shape).
        """
        batch_mean  = x.mean(dim=0)
        batch_var   = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        total_count  = self.count + batch_count
        delta        = batch_mean - self.mean
        new_mean     = self.mean + delta * batch_count / total_count
        m_a          = self.var   * self.count
        m_b          = batch_var  * batch_count
        m2           = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var      = m2 / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the normalised observation, clamped to [-clip, +clip].

        Args:
            x: Tensor of shape (..., *shape).

        Returns:
            Normalised tensor of the same shape.
        """
        normed = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        return normed.clamp(-self.clip, self.clip)

@dataclass
class LagrangianConstraintCfg:
    """
    Defines a single Lagrangian constraint with PID-style dual update.

    The update rule for the log-multiplier is:

        log_λ_{t+1} = log_λ_t
                    + λ_p_gain · violation_t          (proportional)
                    + λ_i_gain · integral_t            (integral)

    where:
        violation_t = J_c(t) - κ       (signed constraint slack)
        integral_t  = Σ_{s≤t} violation_s · dt   (running sum)

    The integral term prevents persistent steady-state violation: if the
    policy consistently operates just above the budget κ, the proportional
    term alone reaches equilibrium with a non-zero residual violation,
    whereas the integral term continues accumulating and drives the
    multiplier higher until the constraint is met exactly.

    Attributes:
        name:          Human-readable label for logging.
        cost_key:      Key into the cost dict returned by the environment.
        budget:        Constraint threshold κ (maximum tolerable expected cost).
        lambda_init:   Initial value for λ (in natural space).
        lambda_p_gain: Proportional gain (equivalent to the former lambda_lr).
        lambda_i_gain: Integral gain. Set to ~10% of lambda_p_gain as a
                       conservative starting point to avoid windup.
        lambda_min:    Lower clamp for λ after exponentiation.
        lambda_max:    Upper clamp for λ after exponentiation.
        integral_clip: Symmetric clamp on the integral accumulator to prevent
                       windup during long periods of unrecoverable violation.
    """
    name:          str
    cost_key:      str
    budget:        float
    lambda_init:   float = 0.1
    lambda_p_gain: float = 5e-4
    lambda_i_gain: float = 5e-5
    lambda_min:    float = 0.0
    lambda_max:    float = 20.0
    integral_clip: float = 100.0


@dataclass
class ConstrainedPPOCfg:
    """
    Algorithm configuration for the PPO-Lagrangian runner.

    Attributes:
        constraints:          List of LagrangianConstraintCfg instances.
        cost_critic_lr:       Learning rate for the cost value network V_c.
        reward_critic_lr:     Learning rate for the reward value network V_r.
        actor_lr:             Learning rate for the policy π.
        clip_param:           PPO clipping parameter ε.
        entropy_coef:         Entropy bonus coefficient c_2.
        value_loss_coef:      Value loss coefficient c_1 (applied to both critics).
        num_learning_epochs:  Number of gradient epochs per rollout batch.
        num_mini_batches:     Number of mini-batches per epoch.
        gamma:                Discount factor for both return estimates.
        lam:                  GAE-λ for advantage estimation.
        desired_kl:           Target KL divergence for adaptive LR scheduling.
        max_grad_norm:        Gradient norm clipping threshold.
    """
    constraints: list[LagrangianConstraintCfg] = field(default_factory=lambda: [
        LagrangianConstraintCfg(
            name="arm_collision",
            cost_key="arm_collision",
            budget=0.05,
            lambda_init=0.1,
            lambda_p_gain=5e-4,
            lambda_max=10.0,
        ),
        LagrangianConstraintCfg(
            name="base_proximity",
            cost_key="base_proximity",
            budget=0.1,
            lambda_init=0.05,
            lambda_p_gain=3e-4,
            lambda_max=15.0,
        ),
    ])
    cost_critic_lr: float = 1e-3
    reward_critic_lr: float = 1e-3
    actor_lr: float = 1e-3
    clip_param: float = 0.2
    entropy_coef: float = 0.015
    value_loss_coef: float = 1.0
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.015
    max_grad_norm: float = 1.0


class ConstrainedActorCritic(nn.Module):
    """
    Actor-Critic network extended with a second value head for cost returns.

    Architecture:
        - Shared observation encoder (none by default; separate MLP trunks).
        - Actor:         MLP → Gaussian policy head (mean + log_std).
        - Reward Critic: Independent MLP → V_r(s) scalar.
        - Cost Critic:   Independent MLP → V_c(s) scalar per constraint.

    The two critics are deliberately kept independent (no shared trunk) to
    prevent gradient interference between the task and safety objectives,
    which is the standard recommendation in CMDP literature.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_constraints: int,
        hidden_dims: list[int],
        init_noise_std: float = 1.0,
        activation: str = "elu",
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        act_fn = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]

        def build_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
            layers, current = [], in_dim
            for h in hidden_dims:
                layers += [nn.Linear(current, h), act_fn()]
                current = h
            layers.append(nn.Linear(current, out_dim))
            return nn.Sequential(*layers)

        # Policy (actor)
        self.actor = build_mlp(obs_dim, action_dim)
        self.log_std = nn.Parameter(
            torch.ones(action_dim, device=device) * torch.log(
                torch.tensor(init_noise_std)
            )
        )

        # Task reward critic V_r
        self.reward_critic = build_mlp(obs_dim, 1)

        # Safety cost critics V_c — one head per constraint
        # Separate networks allow each multiplier to have an independent
        # gradient signal without coupling through shared parameters.
        self.cost_critics = nn.ModuleList(
            [build_mlp(obs_dim, 1) for _ in range(num_constraints)]
        )

        self.to(device)

    def forward(self, obs: torch.Tensor):
        """Returns the action distribution mean for inference."""
        return self.actor(obs)

    def evaluate(self, obs: torch.Tensor):
        """Returns (action_mean, log_std, V_r, [V_c_0, V_c_1, ...])."""
        v_r = self.reward_critic(obs).squeeze(-1)
        v_cs = [critic(obs).squeeze(-1) for critic in self.cost_critics]
        mean = self.actor(obs)
        # Guard against NaN/Inf in actor output
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1e4, neginf=-1e4)
        # Clamp log_std to prevent std collapse (exp(-5)=0.007) or explosion (exp(2)=7.4)
        clamped_log_std = self.log_std.clamp(-5.0, 2.0)
        return mean, clamped_log_std, v_r, v_cs

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples an action and computes its log-probability."""
        mean = self.actor(obs)
        # Guard against NaN/Inf in actor output
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1e4, neginf=-1e4)
        # Clamp log_std to prevent std collapse or explosion
        clamped_log_std = self.log_std.clamp(-5.0, 2.0)
        std = clamped_log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


# =========================================================================
# 5. Constrained On-Policy Runner
# =========================================================================

class ConstrainedOnPolicyRunner:
    """
    PPO-Lagrangian runner implementing the CMDP training loop.

    The modified policy gradient objective is:

        L_total = L_clip(π) + Σ_i λ_i · L_cost_i + c1 · L_vf - c2 · H(π)

    where:
        L_clip  = standard PPO clipped surrogate objective for task rewards.
        L_cost_i = PPO clipped surrogate objective for the i-th safety cost
                   (sign-flipped because higher cost is undesirable).
        L_vf    = MSE loss summed across reward critic and all cost critics.
        H(π)    = policy entropy bonus for exploration.

    Lagrangian multipliers are updated once per iteration via multiplicative
    dual ascent after the policy update, using the observed mean episode cost.
    """

    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        constrained_cfg: ConstrainedPPOCfg,
        runner_cfg: RslRlOnPolicyRunnerCfg,
        log_dir: str,
        device: str = "cuda",
    ):
        self.env = env
        self.cfg = constrained_cfg
        self.runner_cfg = runner_cfg
        self.log_dir = log_dir
        self.device = device
        self.num_constraints = len(constrained_cfg.constraints)
        self.writer = SummaryWriter(log_dir=log_dir)

        # Infer dimensions from the wrapped environment. Some wrapper chains
        # may not expose `observation_space` directly (None); fall back to
        # the unwrapped env's `single_observation_space` and use
        # `gym.spaces.flatdim` to get the flattened vector size.
        # Prefer the unwrapped single-observation/action spaces to avoid
        # accidentally using the already-batched `observation_space` which
        # includes `num_envs` in its shape and would inflate dimensions by N.
        if hasattr(env.unwrapped, "single_observation_space"):
            obs_space = env.unwrapped.single_observation_space
        else:
            obs_space = env.observation_space
        obs_dim = gym.spaces.flatdim(obs_space)
        self.obs_normalizer = RunningMeanStd(shape=(obs_dim,), device=device)

        if hasattr(env.unwrapped, "single_action_space"):
            action_space = env.unwrapped.single_action_space
        else:
            action_space = env.action_space
        action_dim = gym.spaces.flatdim(action_space)

        # --- PID integral accumulators (one scalar per constraint) -----------
        # Stored as plain tensors rather than nn.Parameters because they are
        # updated in a no_grad context and must not appear in the actor's
        # computational graph.
        self.lambda_integrals = [
            torch.tensor(0.0, device=device)
            for _ in constrained_cfg.constraints
        ]

        # --- Actor-Critic with dual critics ----------------------------------
        self.actor_critic = ConstrainedActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_constraints=self.num_constraints,
            hidden_dims=runner_cfg.policy.actor_hidden_dims,
            init_noise_std=runner_cfg.policy.init_noise_std,
            activation=runner_cfg.policy.activation,
            device=device,
        )

        # --- Separate optimizers for actor and each critic -------------------
        # Decoupled optimizers are essential: the Lagrangian loss term must
        # not backpropagate through the cost critics (only through the actor),
        # and independent LRs allow fine-tuning stability per component.
        self.actor_optimizer = optim.Adam(
            self.actor_critic.actor.parameters(),
            lr=constrained_cfg.actor_lr,
        )
        self.reward_critic_optimizer = optim.Adam(
            self.actor_critic.reward_critic.parameters(),
            lr=constrained_cfg.reward_critic_lr,
        )
        self.cost_critic_optimizers = [
            optim.Adam(
                self.actor_critic.cost_critics[i].parameters(),
                lr=constrained_cfg.cost_critic_lr,
            )
            for i in range(self.num_constraints)
        ]

        # --- Lagrangian multipliers (log-space for positivity) ---------------
        # Storing in log-space ensures λ_i is always non-negative after
        # exponentiation, eliminating the need for explicit clamping during
        # the update step itself (clamping is applied only at readout).
        self.log_lambdas = nn.ParameterList([
            nn.Parameter(
                torch.tensor(
                    torch.log(torch.tensor(c.lambda_init)).item(),
                    device=device
                )
            )
            for c in constrained_cfg.constraints
        ])

        # --- Rollout storage -------------------------------------------------
        # num_steps_per_env × num_envs transitions buffered per iteration.
        self.num_steps = runner_cfg.num_steps_per_env
        self.num_envs = env.num_envs

        # Storage buffers — raw tensors rather than a formal RolloutStorage
        # subclass to maintain full control over the cost return computation.
        self._init_storage(obs_dim, action_dim)

        # Iteration counter and best-model tracking
        self.current_iteration = 0
        self.best_mean_reward = -float("inf")

    def _init_storage(self, obs_dim: int, action_dim: int) -> None:
        """Pre-allocates rollout buffers on the target device."""
        T, N = self.num_steps, self.num_envs
        self.obs_buf        = torch.zeros((T, N, obs_dim),  device=self.device)
        self.action_buf     = torch.zeros((T, N, action_dim), device=self.device)
        self.log_prob_buf   = torch.zeros((T, N),            device=self.device)
        self.reward_buf     = torch.zeros((T, N),            device=self.device)
        self.done_buf       = torch.zeros((T, N),            device=self.device)
        self.value_r_buf    = torch.zeros((T, N),            device=self.device)
        # Cost buffers: one channel per constraint
        self.cost_bufs      = torch.zeros((T, N, self.num_constraints), device=self.device)
        self.value_c_bufs   = torch.zeros((T, N, self.num_constraints), device=self.device)

    # -------------------------------------------------------------------------
    # Rollout Collection
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def collect_rollout(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Collects `num_steps_per_env` transitions from all environments.

        Returns the final observation for bootstrapping the last value estimate.
        """
        for step in range(self.num_steps):
            # Sanitize observations to avoid NaNs/Infs flowing into the policy
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, device=self.device)
            obs = obs.to(dtype=torch.float32, device=self.device)
            obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

            actions, log_probs = self.actor_critic.act(obs)
            _, _, v_r, v_cs = self.actor_critic.evaluate(obs)

            # Step the environment — RslRlVecEnvWrapper returns (obs_td, rew, dones, extras).
            next_obs, rewards, dones, extras = self.env.step(actions)

            # Compute costs directly from sensors via the cost functions.
            # The IsaacLab reward_manager only processes the `rewards` config;
            # our `costs` config is NOT evaluated by any built-in manager,
            # so extras["log"] never contains cost keys. We call the cost
            # functions ourselves against the live scene state.
            raw_env = self.env.unwrapped  # Go2RetrievalEnv with scene access
            costs_dict: dict[str, torch.Tensor] = {}
            try:
                costs_dict["arm_collision"] = arm_collision_cost(
                    raw_env, sensor_name="arm_contact", force_threshold=1.0,
                )
            except Exception:
                costs_dict["arm_collision"] = torch.zeros(self.num_envs, device=self.device)
            try:
                costs_dict["base_proximity"] = base_proximity_cost(
                    raw_env, sensor_name="lidar", d_safe=0.4,
                )
            except Exception:
                costs_dict["base_proximity"] = torch.zeros(self.num_envs, device=self.device)
            # If the environment returns a TensorDict (RSL-RL wrapper), convert it.
            if isinstance(next_obs, TensorDict):
                next_obs = self._td_to_tensor(next_obs).to(self.device)
            if isinstance(next_obs, torch.Tensor):
                next_obs = next_obs.to(dtype=torch.float32, device=self.device)
                next_obs = torch.nan_to_num(next_obs, nan=0.0, posinf=1e6, neginf=-1e6)

            self.obs_buf[step]      = obs
            self.action_buf[step]   = actions
            self.log_prob_buf[step] = log_probs
            self.reward_buf[step]   = rewards.squeeze(-1)
            self.done_buf[step]     = dones.float()
            self.value_r_buf[step]  = v_r

            for i, constraint in enumerate(self.cfg.constraints):
                self.cost_bufs[step, :, i]  = costs_dict[constraint.cost_key].squeeze(-1)
                self.value_c_bufs[step, :, i] = v_cs[i]

            obs = next_obs

        return obs

    def _td_to_tensor(self, td):
        """Convert a TensorDict or mapping of observation groups to a single flat tensor.

        Prefers the 'policy' key if present. Otherwise concatenates groups in
        sorted key order.
        """
        if isinstance(td, TensorDict):
            if "policy" in td.keys():
                return td["policy"]
            keys = list(td.keys())
            if len(keys) == 1:
                return td[keys[0]]
            vals = [td[k] for k in sorted(keys)]
            vals = [v.reshape(v.shape[0], -1) for v in vals]
            return torch.cat(vals, dim=-1)
        elif isinstance(td, dict):
            if "policy" in td:
                return torch.as_tensor(td["policy"])
            keys = sorted(td.keys())
            vals = [torch.as_tensor(td[k]) for k in keys]
            vals = [v.reshape(v.shape[0], -1) for v in vals]
            return torch.cat(vals, dim=-1)
        else:
            return td

    # -------------------------------------------------------------------------
    # Return & Advantage Computation
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes GAE-λ advantages and discounted returns.

        Args:
            rewards:    (T, N) reward or cost signal.
            values:     (T, N) value estimates.
            dones:      (T, N) episode termination flags.
            last_value: (N,)   bootstrap value at the final step.

        Returns:
            returns:    (T, N) discounted lambda returns.
            advantages: (T, N) GAE advantages.
        """
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(rewards.shape[1], device=self.device)

        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_val * next_non_terminal - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        return advantages + values, advantages

    # -------------------------------------------------------------------------
    # PPO-Lagrangian Update
    # -------------------------------------------------------------------------

    def update(self, last_obs: torch.Tensor) -> dict:
        """
        Performs the primal (policy + critics) and PID dual (λ) updates.

        Observation normalizer statistics are refreshed from the rollout
        buffer at the start of each update, prior to any gradient computation,
        so that the normalised values used in the mini-batch loop reflect the
        current iteration's data distribution.
        """
        # --- Refresh normalizer from this iteration's rollout data -----------
        # Flatten to (T*N, obs_dim) before updating to avoid dimension mismatch.
        flat_obs = self.obs_buf.reshape(-1, self.obs_buf.shape[-1])
        self.obs_normalizer.update(flat_obs)
        flat_obs_normed = self.obs_normalizer.normalize(flat_obs)

        # Bootstrap values using normalised last observation
        last_obs_normed = self.obs_normalizer.normalize(last_obs)
        with torch.no_grad():
            _, _, last_v_r, last_v_cs = self.actor_critic.evaluate(last_obs_normed)

        # --- GAE computation (unchanged, operates on raw reward/cost buffers) -
        reward_returns, reward_adv = self._compute_gae(
            self.reward_buf, self.value_r_buf, self.done_buf, last_v_r
        )
        cost_returns_list, cost_adv_list = [], []
        for i in range(self.num_constraints):
            c_ret, c_adv = self._compute_gae(
                self.cost_bufs[:, :, i],
                self.value_c_bufs[:, :, i],
                self.done_buf,
                last_v_cs[i],
            )
            cost_returns_list.append(c_ret)
            cost_adv_list.append(c_adv)

        # Flatten remaining buffers (obs already flattened above)
        flat         = lambda t: t.reshape(-1, *t.shape[2:])
        action_flat      = flat(self.action_buf)
        log_prob_old_flat = flat(self.log_prob_buf)
        _raw_adv         = flat(reward_adv)
        _raw_adv         = torch.nan_to_num(_raw_adv, nan=0.0)
        reward_adv_flat  = (_raw_adv - _raw_adv.mean()) / (_raw_adv.std() + 1e-8)
        reward_ret_flat  = torch.nan_to_num(flat(reward_returns), nan=0.0)
        cost_adv_flats   = [torch.nan_to_num(flat(ca), nan=0.0) for ca in cost_adv_list]
        cost_ret_flats   = [torch.nan_to_num(flat(cr), nan=0.0) for cr in cost_returns_list]

        # --- Save pre-update policy distribution for KL computation ---------
        with torch.no_grad():
            pre_mean, pre_log_std, _, _ = self.actor_critic.evaluate(flat_obs_normed)
            self._pre_update_std  = pre_log_std.exp().expand_as(pre_mean).clone()
            self._pre_update_mean = pre_mean.clone()

        # --- Mini-batch PPO updates (normalised observations throughout) -----
        batch_size = flat_obs_normed.shape[0]
        mini_size  = batch_size // self.cfg.num_mini_batches
        metrics    = {"policy_loss": 0., "value_r_loss": 0., "entropy": 0.}
        for i in range(self.num_constraints):
            metrics[f"value_c_loss_{i}"]     = 0.
            metrics[f"constrained_loss_{i}"] = 0.

        for _ in range(self.cfg.num_learning_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_size):
                idx = indices[start: start + mini_size]

                # Use pre-normalised observations — no re-normalisation inside
                # the mini-batch loop, which would introduce stale statistics.
                obs_mb          = flat_obs_normed[idx]
                action_mb       = action_flat[idx]
                log_prob_old_mb = log_prob_old_flat[idx]
                adv_r_mb        = reward_adv_flat[idx]
                ret_r_mb        = reward_ret_flat[idx]

                mean, log_std, v_r, v_cs = self.actor_critic.evaluate(obs_mb)
                std  = log_std.exp().expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                log_prob_new = dist.log_prob(action_mb).sum(dim=-1)
                entropy      = dist.entropy().sum(dim=-1).mean()

                # Clamp log-ratio to prevent extreme ratios (exp(±20) ≈ 5e8)
                log_ratio = (log_prob_new - log_prob_old_mb).clamp(-20.0, 20.0)
                ratio     = log_ratio.exp()
                surr1_r = ratio * adv_r_mb
                surr2_r = ratio.clamp(
                    1 - self.cfg.clip_param, 1 + self.cfg.clip_param
                ) * adv_r_mb
                policy_loss = -torch.min(surr1_r, surr2_r).mean()

                total_cost_loss = torch.tensor(0., device=self.device)
                for i, constraint in enumerate(self.cfg.constraints):
                    adv_c_mb = cost_adv_flats[i][idx]
                    surr1_c  = ratio * adv_c_mb
                    surr2_c  = ratio.clamp(
                        1 - self.cfg.clip_param, 1 + self.cfg.clip_param
                    ) * adv_c_mb
                    lam_i    = self.log_lambdas[i].exp().detach()
                    total_cost_loss = total_cost_loss + lam_i * torch.min(surr1_c, surr2_c).mean()

                vf_loss_total = nn.functional.mse_loss(v_r, ret_r_mb)
                for i in range(self.num_constraints):
                    vf_loss_total = vf_loss_total + nn.functional.mse_loss(
                        v_cs[i], cost_ret_flats[i][idx]
                    )

                total_loss = (
                    policy_loss
                    + total_cost_loss
                    + self.cfg.value_loss_coef * vf_loss_total
                    - self.cfg.entropy_coef * entropy
                )

                self.actor_optimizer.zero_grad()
                self.reward_critic_optimizer.zero_grad()
                for opt in self.cost_critic_optimizers:
                    opt.zero_grad()
                total_loss.backward()

                # Detect NaN gradients — skip this mini-batch if found
                _has_nan_grad = False
                for p in self.actor_critic.parameters():
                    if p.grad is not None and torch.isnan(p.grad).any():
                        _has_nan_grad = True
                        break
                if _has_nan_grad:
                    log.warning("NaN gradient detected — skipping this mini-batch update.")
                    self.actor_optimizer.zero_grad()
                    self.reward_critic_optimizer.zero_grad()
                    for opt in self.cost_critic_optimizers:
                        opt.zero_grad()
                    continue

                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.cfg.max_grad_norm
                )
                self.actor_optimizer.step()
                self.reward_critic_optimizer.step()
                for opt in self.cost_critic_optimizers:
                    opt.step()

                metrics["policy_loss"]  += policy_loss.item()
                metrics["value_r_loss"] += nn.functional.mse_loss(v_r, ret_r_mb).item()
                metrics["entropy"]      += entropy.item()

        n_updates = self.cfg.num_learning_epochs * self.cfg.num_mini_batches
        for k in metrics:
            metrics[k] /= n_updates

        # KL-based adaptive learning rate
        # Compare the pre-update policy (saved before mini-batch loop)
        # against the post-update policy to get a meaningful KL signal.
        with torch.no_grad():
            new_mean, new_log_std, _, _ = self.actor_critic.evaluate(flat_obs_normed)
            new_std = new_log_std.exp().expand_as(new_mean)
            kl = torch.distributions.kl_divergence(
                torch.distributions.Normal(self._pre_update_mean, self._pre_update_std),
                torch.distributions.Normal(new_mean, new_std),
            ).sum(-1).mean()
        metrics["kl"] = kl.item()
        if kl > self.cfg.desired_kl * 2:
            for pg in self.actor_optimizer.param_groups:
                pg["lr"] = max(1e-5, pg["lr"] / 1.5)
        elif kl < self.cfg.desired_kl / 2:
            for pg in self.actor_optimizer.param_groups:
                pg["lr"] = min(1e-2, pg["lr"] * 1.5)

        # =========================================================================
        # PID Dual Ascent: Lagrangian Multiplier Update
        # =========================================================================
        #
        # Full update rule applied in log-space:
        #
        #   violation_t     = J_c(t) - κ
        #   integral_t     += violation_t                       (accumulated)
        #   integral_t      = clamp(integral_t, -I_clip, +I_clip)
        #   log_λ_{t+1}    += P_gain · violation_t
        #                   + I_gain · integral_t
        #   λ_{t+1}         = clamp(exp(log_λ_{t+1}), λ_min, λ_max)
        #
        # The integral accumulator is clamped to prevent windup during extended
        # periods of constraint violation that the policy cannot immediately
        # recover from (e.g., the first few hundred iterations of training when
        # the arm has not yet learned to avoid contact).

        lambda_metrics = {}
        with torch.no_grad():
            for i, constraint in enumerate(self.cfg.constraints):
                mean_cost   = self.cost_bufs[:, :, i].mean().item()
                violation   = mean_cost - constraint.budget

                # Integral accumulation with anti-windup clamp
                self.lambda_integrals[i] = torch.clamp(
                    self.lambda_integrals[i] + violation,
                    -constraint.integral_clip,
                    constraint.integral_clip,
                )

                # PID update in log-space
                self.log_lambdas[i].data += (
                    constraint.lambda_p_gain * violation
                    + constraint.lambda_i_gain * self.lambda_integrals[i].item()
                )

                # Natural-space clamp via log-space bounds
                log_min = torch.log(torch.tensor(constraint.lambda_min + 1e-8)).item()
                log_max = torch.log(torch.tensor(constraint.lambda_max)).item()
                self.log_lambdas[i].data.clamp_(log_min, log_max)

                lam_val = self.log_lambdas[i].exp().item()
                lambda_metrics[f"lambda/{constraint.name}"]          = lam_val
                lambda_metrics[f"lambda/integral_{constraint.name}"] = self.lambda_integrals[i].item()
                lambda_metrics[f"cost/mean_{constraint.name}"]       = mean_cost
                lambda_metrics[f"cost/violation_{constraint.name}"]  = violation

        metrics.update(lambda_metrics)
        return metrics




    # -------------------------------------------------------------------------
    # Checkpoint I/O
    # -------------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "iteration":        self.current_iteration,
            "actor_critic":     self.actor_critic.state_dict(),
            "obs_normalizer":   self.obs_normalizer.state_dict(),
            "log_lambdas":      [lam.data for lam in self.log_lambdas],
            "lambda_integrals": [acc.clone() for acc in self.lambda_integrals],
            "actor_opt":        self.actor_optimizer.state_dict(),
        }, path)


    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(ckpt["actor_critic"])
        self.obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
        for i, lam_data in enumerate(ckpt["log_lambdas"]):
            self.log_lambdas[i].data = lam_data
        for i, acc in enumerate(ckpt["lambda_integrals"]):
            self.lambda_integrals[i] = acc.to(self.device)
        self.actor_optimizer.load_state_dict(ckpt["actor_opt"])
        self.current_iteration = ckpt["iteration"]
        log.info(f"Resumed from iteration {self.current_iteration} at {path}")
    
    # -------------------------------------------------------------------------
    # Main Training Loop
    # -------------------------------------------------------------------------

    def learn(self, num_iterations: int) -> None:
        obs_td, _ = self.env.reset()
        obs = self._td_to_tensor(obs_td).to(self.device)

        for iteration in range(self.current_iteration, num_iterations):
            iter_start = time.perf_counter()

            # --- Curriculum update (before rollout so new threshold is active) --
            new_thresh = env_manager.update_grasp_curriculum(
                env=self.env.unwrapped,          # unwrap to reach Go2RetrievalEnv
                current_iteration=iteration,
                max_iterations=num_iterations,
            )
            self.writer.add_scalar("curriculum/grasp_threshold", new_thresh, iteration)

            # --- Rollout collection ---------------------------------------------
            with torch.no_grad():
                last_obs = self.collect_rollout(obs)
            obs = last_obs

            # --- Primal + dual updates -----------------------------------------
            metrics = self.update(last_obs)
            self.current_iteration = iteration + 1

            for key, val in metrics.items():
                self.writer.add_scalar(key, val, iteration)

            if iteration % 10 == 0:
                elapsed   = time.perf_counter() - iter_start
                lam_strs  = "  ".join(
                    f"λ_{c.name}={metrics[f'lambda/{c.name}']:.4f} "
                    f"∫={metrics[f'lambda/integral_{c.name}']:+.2f} "
                    f"J_c={metrics[f'cost/mean_{c.name}']:.4f}"
                    for c in self.cfg.constraints
                )
                print(
                    f"Iter {iteration:5d} | "
                    f"ε_grasp={new_thresh:.3f}m  "
                    f"π_loss={metrics['policy_loss']:+.4f}  "
                    f"H={metrics['entropy']:.3f}  "
                    f"KL={metrics['kl']:.5f}  "
                    f"dt={elapsed:.2f}s | {lam_strs}",
                    flush=True
                )

            if (iteration + 1) % self.runner_cfg.save_interval == 0:
                ckpt_path = os.path.join(
                    self.log_dir, "checkpoints", f"ckpt_{iteration + 1:05d}.pt"
                )
                self.save(ckpt_path)
                print(f"Checkpoint saved → {ckpt_path}", flush=True)

        self.writer.close()
        print("Training complete.", flush=True)

# =========================================================================
# 6. Runner & Algorithm Configuration
# =========================================================================

def get_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Constructs the RSL-RL base runner configuration."""
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=args_cli.max_iterations,
        save_interval=50,
        experiment_name="go2_retrieval_constrained",
        empirical_normalization=True,
        policy=RslRlPpoActorCriticCfg(
            class_name="ActorCritic",
            init_noise_std=1.0,
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[256, 128, 64],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.015,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.015,
            max_grad_norm=1.0,
        ),
    )

def get_constrained_cfg() -> ConstrainedPPOCfg:
    """
    Constructs the PPO-Lagrangian specific configuration.

    Budget values are set conservatively:
      - arm_collision_budget=0.05 enforces that arm contacts occur in fewer
        than 5% of timesteps at convergence.
      - base_proximity_budget=0.1 permits a small aggregate hinge-loss value
        corresponding to occasional near-misses during gait transitions.
    """
    return ConstrainedPPOCfg(
        constraints=[
            LagrangianConstraintCfg(
                name="arm_collision",
                cost_key="arm_collision",
                budget=0.05,
                lambda_init=0.1,
                lambda_p_gain=5e-4,      # 对应原 lambda_lr
                lambda_i_gain=5e-5,      # 新增积分增益（设为 P 增益的 1/10）
                lambda_min=0.0,
                lambda_max=10.0,
                # integral_clip 使用默认值 100.0，可省略
            ),
            LagrangianConstraintCfg(
                name="base_proximity",
                cost_key="base_proximity",
                budget=0.1,
                lambda_init=0.05,
                lambda_p_gain=3e-4,
                lambda_i_gain=3e-5,
                lambda_min=0.0,
                lambda_max=15.0,
            ),
        ],
        cost_critic_lr=1e-3,
        reward_critic_lr=1e-3,
        actor_lr=1e-3,
        clip_param=0.2,
        entropy_coef=0.015,
        value_loss_coef=1.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        gamma=0.99,
        lam=0.95,
        desired_kl=0.015,
        max_grad_norm=1.0,
    )

# =========================================================================
# 7. Main Entry Point
# =========================================================================

def main():
    # --- Stage 0: Activate experiment configuration -------------------------
    active_exp = experiment_cfg.configure(
        track=args_cli.track,
        group=args_cli.group,
    )
    log.info(f"Active experiment: {active_exp}")

    # --- Environment configuration ------------------------------------------
    env_cfg = Go2RetrievalEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    runner_cfg       = get_runner_cfg()
    constrained_cfg  = get_constrained_cfg()

    # Multi-GPU distribution
    device = "cuda"
    if args_cli.distributed:
        device = f"cuda:{app_launcher.local_rank}"
        env_cfg.sim.device   = device
        runner_cfg.device    = device
        env_cfg.seed        += app_launcher.local_rank

    # --- Log directory ------------------------------------------------------
    log_root = os.path.abspath(
        os.path.join("logs", "ppo_lagrangian", runner_cfg.experiment_name)
    )
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir  = os.path.join(log_root, run_name)
    os.makedirs(log_dir, exist_ok=True)
    log.info(f"Logging to: {log_dir}")

    # --- Environment construction -------------------------------------------
    env = gym.make(
        "Isaac-Go2-Retrieval-v0",
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # --- Stage 0: Dry-run smoke test ----------------------------------------
    if args_cli.dry_run:
        import sys
        log.info("=== DRY-RUN MODE ===")
        print(f"ActiveExperiment: {active_exp}")
        try:
            obs, _ = env.reset()
            action_space = env.action_space
            print(f"Observation shape: {obs['policy'].shape if isinstance(obs, dict) else obs.shape}")
            print(f"Action space:      {action_space}")
            for step_i in range(10):
                action = torch.tensor(
                    action_space.sample(), device="cuda"
                ).unsqueeze(0).expand(args_cli.num_envs, -1)
                obs, reward, terminated, truncated, info = env.step(action)
                obs_shape = obs['policy'].shape if isinstance(obs, dict) else obs.shape
                print(
                    f"  step {step_i:2d} | obs {obs_shape} | "
                    f"reward {reward.mean().item():+.4f}"
                )
            print("Dry-run completed successfully.")
        finally:
            env.close()
        sys.exit(0)

    if args_cli.video:
        video_kwargs = {
            "video_folder":   os.path.join(log_dir, "videos", "train"),
            "step_trigger":   lambda step: step % args_cli.video_interval == 0,
            "video_length":   args_cli.video_length,
            "disable_logger": True,
        }
        log.info("Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # --- Wrap for RSL-RL compatibility --------------------------------------
    # RslRlVecEnvWrapper is expected to pass the `costs` dictionary through
    # its step() return as the third element: (obs, reward, costs, done, info).
    # If your version of isaaclab_rl does not yet expose costs natively,
    # subclass RslRlVecEnvWrapper and override step() to extract `info["costs"]`.
    env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions if hasattr(runner_cfg, 'clip_actions') else None)

    # --- Persist configurations for reproducibility -------------------------
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "runner.yaml"), runner_cfg)

    # --- Instantiate constrained runner -------------------------------------
    runner = ConstrainedOnPolicyRunner(
        env=env,
        constrained_cfg=constrained_cfg,
        runner_cfg=runner_cfg,
        log_dir=log_dir,
        device=device,
    )

    # Optionally resume from a prior checkpoint
    if args_cli.resume_path is not None:
        runner.load(args_cli.resume_path)

    # --- Training -----------------------------------------------------------
    start = time.time()
    runner.learn(num_iterations=args_cli.max_iterations)
    log.info(f"Total training time: {round(time.time() - start, 1)} s")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()