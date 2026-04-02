"""
Differentiable SDF guidance for arm collision avoidance.
Filename: sdf_guidance.py

Builds a voxel-grid SDF from the 3 cylinder obstacles every episode reset
and applies a zero-shot gradient correction to the raw diffusion output.

No PhysX coupling — the SDF lives entirely in a GPU tensor buffer.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


# =============================================================================
# Z1 forward-kinematics stub  (DH parameters from the Z1 URDF)
# =============================================================================

# DH parameters: [a, alpha, d, theta_offset] for links 1-6
# Values taken from the Unitree Z1 URDF (approximate).
_Z1_DH: list[tuple[float, float, float, float]] = [
    (0.0,    math.pi / 2,  0.0,    0.0),   # joint1
    (0.35,   0.0,          0.0,    0.0),   # joint2
    (0.345,  0.0,          0.0,    0.0),   # joint3
    (0.0,    math.pi / 2,  0.0,    0.0),   # joint4
    (0.0,   -math.pi / 2,  0.0,    0.0),   # joint5
    (0.0,    0.0,          0.084,  0.0),   # joint6 (to EE)
]


def _dh_transform(a: float, alpha: float, d: float, theta: torch.Tensor) -> torch.Tensor:
    """
    Compute the 4×4 DH homogeneous transform for a batch of joint angles.

    Args:
        theta: (B,) joint angles in radians.
    Returns:
        T: (B, 4, 4) homogeneous transforms.
    """
    B = theta.shape[0]
    device = theta.device
    ct = theta.cos()
    st = theta.sin()
    ca = math.cos(alpha)
    sa = math.sin(alpha)

    T = torch.zeros(B, 4, 4, device=device)
    T[:, 0, 0] = ct
    T[:, 0, 1] = -st * ca
    T[:, 0, 2] = st * sa
    T[:, 0, 3] = a * ct
    T[:, 1, 0] = st
    T[:, 1, 1] = ct * ca
    T[:, 1, 2] = -ct * sa
    T[:, 1, 3] = a * st
    T[:, 2, 1] = sa
    T[:, 2, 2] = ca
    T[:, 2, 3] = d
    T[:, 3, 3] = 1.0
    return T


def z1_forward_kinematics(
    q: torch.Tensor,
    base_T: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute world-frame link-tip positions for all 6 Z1 joints.

    Args:
        q:      (B, 6) joint angles.
        base_T: (B, 4, 4) world-frame base transform of the arm root.
                Defaults to identity if None.

    Returns:
        positions: (B, 6, 3) world-frame XYZ of each link tip.
    """
    B = q.shape[0]
    device = q.device

    if base_T is None:
        T_accum = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    else:
        T_accum = base_T.clone()

    link_positions = torch.zeros(B, 6, 3, device=device)

    for i, (a, alpha, d, theta_off) in enumerate(_Z1_DH):
        theta = q[:, i] + theta_off
        T_i = _dh_transform(a, alpha, d, theta)          # (B, 4, 4)
        T_accum = torch.bmm(T_accum, T_i)
        link_positions[:, i, :] = T_accum[:, :3, 3]     # tip XYZ

    return link_positions                                 # (B, 6, 3)


# =============================================================================
# Voxel-grid SDF builder
# =============================================================================

class VoxelSDF:
    """
    GPU-resident axis-aligned voxel-grid signed-distance field.

    The SDF is unsigned inside obstacles (negative sign convention not needed
    for hinge-loss collision energy) and built from cylinder primitives.
    """

    def __init__(
        self,
        grid_res: int,
        world_min: tuple[float, float, float],
        world_max: tuple[float, float, float],
        device: str | torch.device,
    ) -> None:
        self.grid_res = grid_res
        self.world_min = torch.tensor(world_min, device=device)
        self.world_max = torch.tensor(world_max, device=device)
        self.device = device

        # Voxel spacing
        self.voxel_size = (self.world_max - self.world_min) / grid_res

        # Grid buffer: (1, 1, R, R, R) for F.grid_sample
        self._grid: torch.Tensor | None = None

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------

    def build_from_cylinders(
        self,
        centers: torch.Tensor,   # (K, 3) — XY footprint + half-height in Z
        radii: torch.Tensor,     # (K,)
        heights: torch.Tensor,   # (K,)
    ) -> None:
        """
        Populate the SDF grid from K vertical cylinder obstacles.

        Each cylinder is axis-aligned with the world Z-axis.
        SDF value = minimum distance to any cylinder surface.
        """
        R = self.grid_res
        device = self.device

        # Build 3-D grid of world-frame voxel centres: (R, R, R, 3)
        lin = [
            torch.linspace(
                float(self.world_min[k].item()),
                float(self.world_max[k].item()),
                R,
                device=device,
            )
            for k in range(3)
        ]
        grid_z, grid_y, grid_x = torch.meshgrid(lin[2], lin[1], lin[0], indexing="ij")
        voxels = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (R, R, R, 3)

        # Compute SDF for each cylinder and take the minimum
        sdf = torch.full((R, R, R), fill_value=1e6, device=device)

        K = centers.shape[0]
        for k in range(K):
            cx, cy, cz = centers[k, 0], centers[k, 1], centers[k, 2]
            r = radii[k]
            h = heights[k] * 0.5  # half-height

            dx = voxels[..., 0] - cx
            dy = voxels[..., 1] - cy
            dz = voxels[..., 2] - cz

            # Radial distance in XY plane minus cylinder radius
            radial = torch.sqrt(dx ** 2 + dy ** 2) - r
            # Vertical distance outside the half-height slab
            vertical = torch.abs(dz) - h

            # Exact exterior SDF for a capped cylinder
            outside_r = torch.clamp(radial, min=0.0)
            outside_z = torch.clamp(vertical, min=0.0)
            inside    = torch.clamp(torch.maximum(radial, vertical), max=0.0)
            d_cyl = torch.sqrt(outside_r ** 2 + outside_z ** 2) + inside

            sdf = torch.minimum(sdf, d_cyl)

        # Store in (1, 1, R, R, R) format for F.grid_sample
        self._grid = sdf.unsqueeze(0).unsqueeze(0)

    # -------------------------------------------------------------------------
    # Query — differentiable trilinear interpolation
    # -------------------------------------------------------------------------

    def query(self, points: torch.Tensor) -> torch.Tensor:
        """
        Query the SDF at arbitrary world-frame points via trilinear interpolation.

        Args:
            points: (..., 3) world-frame XYZ.
        Returns:
            sdf_values: (...,) signed distances (positive = outside all obstacles).
        """
        if self._grid is None:
            raise RuntimeError("SDF grid not built; call build_from_cylinders first.")

        shape = points.shape[:-1]
        pts = points.reshape(1, 1, 1, -1, 3)           # (1, 1, 1, N, 3)

        # Normalise to [-1, 1] for F.grid_sample
        norm = 2.0 * (pts - self.world_min) / (self.world_max - self.world_min) - 1.0

        # F.grid_sample expects grid coords in (x, y, z) order for 3D
        out = F.grid_sample(
            self._grid,                                  # (1, 1, R, R, R)
            norm,                                        # (1, 1, 1, N, 3)
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )                                               # (1, 1, 1, 1, N)
        return out.squeeze().reshape(shape)              # (...,)


# =============================================================================
# SDF Energy + guidance
# =============================================================================

class SDFGuidance:
    """
    Zero-shot SDF collision-energy gradient correction for arm actions.

    Applied after the 1-step diffusion output:
        a_guided = a_raw - alpha * grad_a E_SDF(FK(a_raw))

    The Jacobian is approximated by finite differences to avoid a backward
    pass through the network.
    """

    def __init__(
        self,
        alpha: float,
        d_safe: float,
        fd_eps: float,
        device: str | torch.device,
        grid_res: int = 32,
        world_min: tuple[float, float, float] = (-1.0, -1.5, -0.1),
        world_max: tuple[float, float, float] = (4.0, 1.5, 2.0),
    ) -> None:
        self.alpha = alpha
        self.d_safe = d_safe
        self.fd_eps = fd_eps
        self.device = device

        self._sdf = VoxelSDF(grid_res, world_min, world_max, device)
        self._base_T: torch.Tensor | None = None   # (N, 4, 4) arm base transforms

    # -------------------------------------------------------------------------

    def is_ready(self) -> bool:
        return self._sdf._grid is not None

    def rebuild(
        self,
        centers: torch.Tensor,
        radii: torch.Tensor,
        heights: torch.Tensor,
        base_T: torch.Tensor | None = None,
    ) -> None:
        """
        Rebuild the SDF from new obstacle positions.

        Args:
            centers: (K, 3) obstacle XYZ centres.
            radii:   (K,)   cylinder radii.
            heights: (K,)   cylinder heights.
            base_T:  (N, 4, 4) optional arm base transforms (per-env).
        """
        self._sdf.build_from_cylinders(centers, radii, heights)
        self._base_T = base_T

    # -------------------------------------------------------------------------

    def _energy(self, q: torch.Tensor) -> torch.Tensor:
        """
        Soft hinge collision energy  E = Σ_i max(0, d_safe - d_sdf(p_link_i))^2.

        Args:
            q: (B, 6) joint angles.
        Returns:
            E: (B,) energy per environment.
        """
        link_pos = z1_forward_kinematics(q, self._base_T)  # (B, 6, 3)
        B, L, _ = link_pos.shape

        pts = link_pos.reshape(B * L, 3)
        d = self._sdf.query(pts).reshape(B, L)             # (B, 6)

        violation = torch.clamp(self.d_safe - d, min=0.0)
        return (violation ** 2).sum(dim=-1)                 # (B,)

    def apply_guidance(self, a_raw: torch.Tensor) -> torch.Tensor:
        """
        Compute FD Jacobian of E w.r.t. action and apply gradient step.

        Args:
            a_raw: (B, action_dim) raw diffusion output.
        Returns:
            a_guided: (B, action_dim) corrected action.
        """
        B, D = a_raw.shape
        grad = torch.zeros_like(a_raw)

        E0 = self._energy(a_raw)                           # (B,)

        for d in range(D):
            a_plus = a_raw.clone()
            a_plus[:, d] += self.fd_eps
            E_plus = self._energy(a_plus)
            grad[:, d] = (E_plus - E0) / self.fd_eps

        return a_raw - self.alpha * grad


# =============================================================================
# Environment-level rebuild helper (called from event_cfg reset)
# =============================================================================

def rebuild_sdf(env, env_ids: torch.Tensor) -> None:
    """
    Rebuild the shared SDF from current obstacle world poses.

    Called at the end of reset_obstacles_curriculum_safe so that the SDF
    is always consistent with the current episode's obstacle layout.
    This function is a no-op if ConsistencyArmPolicy is not loaded.
    """
    # Locate the arm action term in the action manager
    arm_term = None
    try:
        action_manager = env.action_manager
        for term in action_manager._terms.values():
            if hasattr(term, "_sdf_guidance") and term._sdf_guidance is not None:
                arm_term = term
                break
    except Exception:
        return

    if arm_term is None:
        return

    sdf_guidance = arm_term._sdf_guidance
    device = env.device

    # Collect obstacle positions (use only the first env_id for shared SDF)
    obstacle_names = ["obstacle_0", "obstacle_1", "obstacle_2"]
    centers_list, radii_list, heights_list = [], [], []

    for name in obstacle_names:
        try:
            obs = env.scene[name]
            # Take positions from env_ids[0] as representative for all envs
            pos = obs.data.root_pos_w[env_ids[:1]]  # (1, 3)
            centers_list.append(pos.squeeze(0))
            radii_list.append(torch.tensor(0.15, device=device))
            heights_list.append(torch.tensor(0.5, device=device))
        except KeyError:
            pass

    if not centers_list:
        return

    centers = torch.stack(centers_list, dim=0)   # (K, 3)
    radii   = torch.stack(radii_list, dim=0)     # (K,)
    heights = torch.stack(heights_list, dim=0)   # (K,)

    # Arm base transform: identity for now (mount is at robot base)
    sdf_guidance.rebuild(centers, radii, heights, base_T=None)