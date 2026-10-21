"""Shared single-device test fixtures, extracted without distributed test dependencies."""
import warnings
from contextlib import contextmanager
import torch
from fastmd.models.matris.model import MatRIS

def toy_model(*, smoothed=False, norm="rms", reference="demo", angular="fourier", dtype=torch.float32):
    torch.manual_seed(47)
    model = MatRIS(num_layers=2, node_feat_dim=6, edge_feat_dim=6,
                   three_body_feat_dim=6, mlp_hidden_dims=(6,), num_radial=3,
                   num_angular=3, max_n=2, max_l=2, norm_type=norm,
                   three_body_expansion=angular, pairwise_cutoff=3., three_body_cutoff=2.,
                   reference_energy=reference, use_smoothed_for_delta_edge=smoothed,
                   use_bias=True, enable_compile=False, enable_checkpoint=False).to(dtype=dtype).eval()
    model.requires_grad_(False)
    return model

@contextmanager
def deterministic_algorithms():
    """Run-to-run bitwise CUDA forces (ordered sums instead of atomics); restores the flags."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ...but an op without a deterministic implementation fails the test.
            warnings.filterwarnings("error", message=".*does not have a deterministic implementation.*")
            yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

import math
import numpy as np
from ase import Atoms

def system() -> Atoms:
    rng = np.random.default_rng(123)
    grid = np.stack(np.meshgrid(*(np.arange(4),) * 3, indexing="ij"), -1).reshape(-1, 3)
    atoms = Atoms(
        "H32C32", positions=grid * 1.3 + rng.uniform(-0.1, 0.1, (64, 3)),
        cell=[5.2] * 3, pbc=True,
    )
    atoms.set_momenta(rng.normal(0.0, 0.3, (64, 3)))
    return atoms

def deterministic_forces(model) -> None:
    phase = torch.tensor(
        np.random.default_rng(7).uniform(0.0, 1.0, (4096, 3)),
        dtype=torch.float32, device="cuda",
    )

    def forward(graphs, task="ef", is_training=False, n_real=None, **_):
        frac = graphs[0].atom_frac_coord
        arg = 2 * math.pi * (frac - phase[: frac.shape[0]])
        energy = (0.8 / (2 * math.pi) * torch.cos(arg)).sum().reshape(1)
        return {"e": energy, "f": [0.8 * torch.sin(arg)]}

    model.forward = forward
