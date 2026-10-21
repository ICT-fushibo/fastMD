"""Measured residual spectrum and mass-weighted finite-difference curvature.

These are diagnostics, not universal sufficient conditions for MTS stability.
"""
import numpy as np
from scipy.signal import periodogram
from fastmd.interfaces.mts import ACCELERATION_CONVERSION


def residual_spectrum(residual_forces, masses, sample_interval_fs, outer_steps_fs):
    forces = np.asarray(residual_forces, dtype=float)
    g = (forces / np.sqrt(np.asarray(masses))[None, :, None]).reshape(len(forces), -1)
    frequencies, component_psd = periodogram(g, fs=1/sample_interval_fs, window='hann',
                                            detrend='constant', axis=0, scaling='density')
    psd = np.sum(component_psd, axis=1)
    df = frequencies[1]-frequencies[0]
    total = float(psd.sum()*df)
    rms_omega = float(np.sqrt(np.sum(psd*(2*np.pi*frequencies)**2)/psd.sum())) if total > 0 else 0.0
    metrics = {}
    for H in outer_steps_fs:
        threshold = 1/(2*H)
        high = float(psd[frequencies > threshold].sum()*df)
        metrics[str(H)] = {'eta': high/total if total > 0 else None,
                            'high_power': high, 'cutoff_cycles_per_fs': threshold,
                            'frequency_band_resolved': bool(threshold < frequencies[-1]),
                            'H_omega_rms': H*rms_omega}
    return {'force_rmse_ev_ang': float(np.sqrt(np.mean(forces**2))),
            'mass_weighted_rms': float(np.sqrt(np.mean(g**2))),
            'mean_mass_weighted_force_norm': float(np.linalg.norm(g.mean(axis=0))),
            'total_fluctuation_power': total, 'omega_rms_per_fs': rms_omega,
            'frequency_resolution_per_fs': float(df), 'sample_interval_fs': sample_interval_fs,
            'nframes': len(forces), 'outer': metrics}, frequencies, psd


def directional_curvature(evaluator, positions, masses, basis, max_displacement_ang=1e-3):
    """Return D Q in fs^-2; each column uses its own mass-coordinate epsilon."""
    q = np.asarray(positions)
    invsqrt = 1/np.sqrt(np.repeat(masses, 3))
    columns = []
    for direction in np.asarray(basis).T:
        displacement = (invsqrt*direction).reshape(q.shape)
        epsilon = max_displacement_ang/np.max(np.linalg.norm(displacement, axis=1))
        _, plus = evaluator(q+epsilon*displacement)
        _, minus = evaluator(q-epsilon*displacement)
        columns.append(-invsqrt*(plus-minus).ravel()/(2*epsilon)*ACCELERATION_CONVERSION)
    return np.column_stack(columns)


def curvature_summary(basis, d_times_basis):
    projection = basis.T @ d_times_basis
    asymmetry = np.linalg.norm(projection-projection.T)/max(np.linalg.norm(projection), 1e-30)
    eigenvalues = np.linalg.eigvalsh((projection+projection.T)/2)
    # Singular norm of DQ is a measured lower bound on ||D||, not an upper bound.
    norm = float(np.linalg.norm(d_times_basis, ord=2))
    return {'probe_norm_fs_minus2': norm, 'projected_min_eigenvalue_fs_minus2': float(eigenvalues[0]),
            'projected_max_eigenvalue_fs_minus2': float(eigenvalues[-1]),
            'projection_asymmetry_relative': float(asymmetry), 'directions': basis.shape[1]}


def linear_respa_radius(ds, dd, h, m):
    n = len(ds)
    eye, zero = np.eye(n), np.zeros((n, n))
    def kick(d, t):
        return np.block([[eye, zero], [-t*d, eye]])
    drift = np.block([[eye, h*eye], [zero, eye]])
    inner = kick(ds, h/2) @ drift @ kick(ds, h/2)
    outer = kick(dd, h*m/2)
    update = outer @ np.linalg.matrix_power(inner, m) @ outer
    return float(np.max(np.abs(np.linalg.eigvals(update))))
