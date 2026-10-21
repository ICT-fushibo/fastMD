"""A direct force head may already return ASE stress units (e.g. ORB)."""
import unittest
import torch
from ase import units
from fastmd.adapters.protocol import CaptureContract
from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics


class DirectAdapter:
    def __init__(self, unit):
        self.module = torch.nn.Linear(1, 1)
        self.unit = unit

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def extensive_energy(self, value, count):
        return value

    def capture_contract(self):
        return CaptureContract(sync_free=False, exact_cutoff=False,
                               force_mode='direct_head', stress_unit=self.unit)


class ModelOwnedStressTest(unittest.TestCase):
    def test_logged_pressure_has_same_units_for_gpa_and_ase_models(self):
        expected = torch.eye(3, dtype=torch.float64) * units.GPa
        for unit, raw in [('GPa', torch.eye(3,dtype=torch.float64)), ('eV/A^3', expected)]:
            with self.subTest(unit=unit):
                md = GPUResidentMolecularDynamics.__new__(GPUResidentMolecularDynamics)
                md._adapter = DirectAdapter(unit)
                md.model = md._adapter.module
                md.nsteps = 10
                md._stress_at = (10, raw)
                torch.testing.assert_close(md._logged_stress(), expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
