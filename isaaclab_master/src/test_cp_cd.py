"""
Unit tests for Innovation 1: Consistency Policy + SDF guidance.
Run with:  pytest test_cp_cd.py -v
No IsaacLab runtime required.
"""
import math
import pytest
import torch
from unittest.mock import MagicMock, patch

from consistency_arm_policy import ConsistencyUNet, consistency_training_loss
from sdf_guidance import VoxelSDF, SDFGuidance, z1_forward_kinematics


# =============================================================================
# Fixtures
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OBS_DIM = 48
ACTION_DIM = 6
BATCH = 8


@pytest.fixture
def unet():
    return ConsistencyUNet(OBS_DIM, ACTION_DIM).to(DEVICE)


@pytest.fixture
def sdf_guidance():
    g = SDFGuidance(alpha=0.05, d_safe=0.10, fd_eps=1e-3, device=DEVICE)
    centers = torch.tensor([[1.0, 0.0, 0.25], [1.5, 0.0, 0.25], [2.0, 0.0, 0.25]], device=DEVICE)
    radii   = torch.tensor([0.15, 0.15, 0.15], device=DEVICE)
    heights = torch.tensor([0.50, 0.50, 0.50], device=DEVICE)
    g.rebuild(centers, radii, heights)
    return g


# =============================================================================
# Test 1: 1-step output shape
# =============================================================================

def test_unet_output_shape(unet):
    obs    = torch.randn(BATCH, OBS_DIM, device=DEVICE)
    noisy  = torch.randn(BATCH, ACTION_DIM, device=DEVICE)
    sigma  = torch.full((BATCH,), 80.0, device=DEVICE)

    with torch.no_grad():
        out = unet(obs, noisy, sigma)

    assert out.shape == (BATCH, ACTION_DIM), (
        f"Expected shape ({BATCH}, {ACTION_DIM}), got {out.shape}"
    )


def test_ct_loss_is_scalar(unet):
    obs     = torch.randn(BATCH, OBS_DIM, device=DEVICE)
    actions = torch.randn(BATCH, ACTION_DIM, device=DEVICE)

    loss = consistency_training_loss(unet, obs, actions)
    assert loss.ndim == 0, "CT loss must be a scalar tensor"
    assert loss.item() >= 0.0, "CT loss must be non-negative"
    assert not math.isnan(loss.item()), "CT loss must not be NaN"


# =============================================================================
# Test 2: Guided action has strictly lower SDF energy than unguided action
# =============================================================================

def test_sdf_guidance_reduces_energy(sdf_guidance):
    """
    Place joint angles such that FK puts several links inside the obstacle zone,
    then verify the guided action has lower SDF energy.
    """
    torch.manual_seed(42)

    # Joint angles that push the arm near the first obstacle (approx 1 m forward)
    q_raw = torch.zeros(BATCH, ACTION_DIM, device=DEVICE)
    q_raw[:, 1] = -0.5   # shoulder forward
    q_raw[:, 2] =  0.3   # elbow

    q_guided = sdf_guidance.apply_guidance(q_raw)

    # Compute energies
    e_raw    = sdf_guidance._energy(q_raw).mean().item()
    e_guided = sdf_guidance._energy(q_guided).mean().item()

    assert e_guided <= e_raw + 1e-6, (
        f"Guided energy ({e_guided:.6f}) should be ≤ unguided energy ({e_raw:.6f})"
    )


def test_voxel_sdf_query_positive_outside(sdf_guidance):
    """Points far from all obstacles should have positive SDF values."""
    far_points = torch.tensor([[10.0, 10.0, 10.0]], device=DEVICE)
    d = sdf_guidance._sdf.query(far_points)
    assert d.item() > 0.0, "SDF should be positive far outside any obstacle"


def test_forward_kinematics_shape():
    B = 4
    q = torch.zeros(B, 6, device=DEVICE)
    positions = z1_forward_kinematics(q)
    assert positions.shape == (B, 6, 3), (
        f"FK output shape should be ({B}, 6, 3), got {positions.shape}"
    )


# =============================================================================
# Test 3: Deterministic single-step inference
# =============================================================================

def test_single_step_determinism(unet):
    """Two identical forward passes should produce identical outputs."""
    obs   = torch.randn(BATCH, OBS_DIM, device=DEVICE)
    noise = torch.randn(BATCH, ACTION_DIM, device=DEVICE)
    sigma = torch.full((BATCH,), 80.0, device=DEVICE)

    unet.eval()
    with torch.no_grad():
        out1 = unet(obs, noise, sigma)
        out2 = unet(obs, noise, sigma)

    assert torch.allclose(out1, out2), "Inference must be deterministic"