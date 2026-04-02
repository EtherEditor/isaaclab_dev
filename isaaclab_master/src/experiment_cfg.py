"""
Experiment Configuration — Stage 0 Scaffolding
Filename: experiment_cfg.py
Experiment Group: B0 (Baseline)
Innovation Point: Shared experiment configuration singleton

Provides the experiment-group and innovation-track enums, the
ActiveExperiment dataclass, and a module-level configure() function
that every subsequent stage will import to determine which code
paths to activate.

Usage::

    import experiment_cfg
    exp = experiment_cfg.configure(
        track=experiment_cfg.InnovationTrack.SAMPLING,
        group=experiment_cfg.ExperimentGroup.B0,
    )
    assert experiment_cfg.ACTIVE_EXPERIMENT is exp
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# =========================================================================
# 1. Experiment Group Enum
# =========================================================================

class ExperimentGroup(str, Enum):
    """Experiment group labels from the experiment outline."""
    B0 = "B0"
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"
    E4 = "E4"


# =========================================================================
# 2. Innovation Track Enum
# =========================================================================

class InnovationTrack(str, Enum):
    """Three primary innovation axes of the research."""
    SAMPLING  = "sampling"      # Innovation 1
    SIM2REAL  = "sim2real"      # Innovation 2
    VLA_GRASP = "vla_grasp"     # Innovation 3


# =========================================================================
# 3. Active Experiment Dataclass
# =========================================================================

def _make_run_id(track: InnovationTrack, group: ExperimentGroup) -> str:
    """Generate a deterministic, sortable run identifier."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{track.value}_{group.value}_{ts}"


@dataclass
class ActiveExperiment:
    """
    Immutable (by convention) record of the currently active experiment.

    Attributes:
        track:  The innovation track being evaluated.
        group:  The experiment group (baseline or ablation variant).
        run_id: Auto-generated string of the form
                ``"{track.value}_{group.value}_{YYYYMMDD_HHMMSS}"``.
    """
    track:  InnovationTrack
    group:  ExperimentGroup
    run_id: str = field(default="", init=True)

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = _make_run_id(self.track, self.group)


# =========================================================================
# 4. Module-Level Singleton
# =========================================================================

ACTIVE_EXPERIMENT: Optional[ActiveExperiment] = None


def configure(
    track: InnovationTrack | str,
    group: ExperimentGroup | str,
) -> ActiveExperiment:
    """
    Instantiate and register the active experiment singleton.

    This function is idempotent within a process: calling it multiple times
    overwrites the previous configuration (useful for unit-test parametrisation).

    Args:
        track: An ``InnovationTrack`` member or its string value
               (e.g. ``"sampling"``).
        group: An ``ExperimentGroup`` member or its string value
               (e.g. ``"B0"``).

    Returns:
        The newly created ``ActiveExperiment`` instance, which is also
        accessible via ``experiment_cfg.ACTIVE_EXPERIMENT``.
    """
    global ACTIVE_EXPERIMENT

    # Accept raw strings from argparse for ergonomic CLI usage.
    if isinstance(track, str):
        track = InnovationTrack(track)
    if isinstance(group, str):
        group = ExperimentGroup(group)

    ACTIVE_EXPERIMENT = ActiveExperiment(track=track, group=group)
    return ACTIVE_EXPERIMENT
