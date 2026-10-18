import numpy as np
import pandas as pd

from src.experiments import _block_bootstrap_indices, _find_equilibria


def test_equilibrium_solver_classifies_interior_and_boundary_branches():
    x = np.linspace(0.001, 0.999, 101)
    curves = pd.DataFrame({
        "kappa": 0.0,
        "x": x,
        "mean_dF": x - 0.5,
    })

    equilibria = _find_equilibria(curves)
    interior = equilibria[equilibria["equilibrium_type"] == "interior"].iloc[0]
    boundaries = equilibria[equilibria["equilibrium_type"] == "boundary"]

    assert np.isclose(interior["x_star"], 0.5)
    assert interior["stability"] == "unstable"
    assert boundaries.set_index("x_star")["stability"].to_dict() == {
        0.0: "stable",
        1.0: "stable",
    }


def test_block_bootstrap_returns_requested_days_and_repeats_blocks():
    sample = _block_bootstrap_indices(
        day_count=5, sample_count=20, block_length=2,
        rng=np.random.default_rng(19),
    )

    assert sample.shape == (20,)
    assert np.all((sample >= 0) & (sample < 5))
    assert len(np.unique(sample)) < len(sample)
