"""
Unit tests for Innovation 2: RMA Adaptation Module.
Run with:  pytest test_rma.py -v
No IsaacLab runtime required.
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from adaptation_module import AdaptationModule, AdaptationModuleAdapter


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH  = 16
T      = AdaptationModule.HISTORY_LEN
IN_DIM = AdaptationModule.INPUT_DIM
EMBED  = AdaptationModule.EMBED_DIM


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def module():
    return AdaptationModule().to(DEVICE)


@pytest.fixture
def adapter():
    return AdaptationModuleAdapter(num_envs=BATCH, device=DEVICE)


# =============================================================================
# Test 1: LSTM output shape
# =============================================================================

def test_lstm_output_shape(module):
    history = torch.randn(BATCH, T, IN_DIM, device=DEVICE)
    z_e = module(history)
    assert z_e.shape == (BATCH, EMBED), (
        f"Expected ({BATCH}, {EMBED}), got {z_e.shape}"
    )


def test_z_e_bounded(module):
    """Tanh output should be in [-1, 1]."""
    history = torch.randn(BATCH, T, IN_DIM, device=DEVICE) * 100.0
    z_e = module(history)
    assert z_e.abs().max().item() <= 1.0 + 1e-5, "Tanh output must be in [-1, 1]"


# =============================================================================
# Test 2: z_e correlates with ground-truth arm inertia
# =============================================================================

def test_z_e_correlates_with_inertia(module):
    """
    Generate synthetic privileged state (arm inertia proxy) and verify that
    a trained projection correlates with z_e via Pearson r > 0 after a
    few gradient steps of the distillation loss.
    """
    torch.manual_seed(0)
    priv_dim = 9   # 6 inertia + 3 CoM
    priv_proj = nn.Linear(priv_dim, EMBED, bias=False).to(DEVICE)
    opt = torch.optim.Adam(list(module.parameters()) + list(priv_proj.parameters()), lr=1e-3)

    # Synthetic training: privileged state encodes a scalar inertia scale
    inertia_scales = torch.rand(BATCH, device=DEVICE) * 10.0

    for _ in range(100):
        history = inertia_scales.unsqueeze(-1).unsqueeze(-1).expand(BATCH, T, IN_DIM)
        history = history + 0.01 * torch.randn_like(history)

        priv_state = torch.zeros(BATCH, priv_dim, device=DEVICE)
        priv_state[:, 0] = inertia_scales   # Ixx ~ scale

        z_e = module(history)
        target = priv_proj(priv_state.detach())
        loss = 0.1 * torch.nn.functional.mse_loss(z_e, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

    # Check Pearson correlation between first z_e dim and inertia scale
    with torch.no_grad():
        z_e_final = module(history)[:, 0].cpu()
        scale_cpu  = inertia_scales.cpu()

    z_mean = z_e_final.mean()
    s_mean = scale_cpu.mean()
    r_num  = ((z_e_final - z_mean) * (scale_cpu - s_mean)).sum()
    r_den  = (z_e_final - z_mean).std() * (scale_cpu - s_mean).std() * BATCH
    r      = (r_num / (r_den + 1e-8)).item()

    assert r > 0.0, f"Expected positive Pearson r, got r={r:.4f}"


# =============================================================================
# Test 3: Adapter step / reset preserves GPU residency
# =============================================================================

def test_adapter_device(adapter):
    assert adapter.z_e.device.type == DEVICE.split(":")[0]
    assert adapter._history.device.type == DEVICE.split(":")[0]


def test_adapter_reset_partial(adapter):
    adapter._history[:] = 1.0
    env_ids = torch.tensor([0, 1, 2], device=DEVICE)
    adapter.reset(env_ids)
    assert adapter._history[:3].abs().sum().item() == 0.0
    assert adapter._history[3:].abs().sum().item() > 0.0


def test_adapter_step_updates_ze(adapter):
    """After one step with a mock env, z_e should differ from the initial zeros."""
    mock_env = MagicMock()
    mock_env.scene = MagicMock()

    # Mock robot applied_torque
    mock_robot = MagicMock()
    mock_robot.data.applied_torque = torch.randn(BATCH, 20, device=DEVICE)
    mock_env.scene.__getitem__ = MagicMock(return_value=mock_robot)

    # Mock contact sensor
    mock_sensor = MagicMock()
    mock_sensor.data.net_forces_w = torch.randn(BATCH, 10, 3, device=DEVICE)
    mock_env.scene.sensors = {"base_contact": mock_sensor}

    z_before = adapter.z_e.clone()
    adapter.step(mock_env)
    assert not torch.allclose(adapter.z_e, z_before), "z_e should change after a step"