"""Extraction boundaries and end-to-end CPU checks with no pretrained downloads."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from ase import Atoms
from ase.io import read

from fastmd import InferenceConfig, MLIPCalculator, MolecularDynamics
from fastmd.adapters import load_adapter
from fastmd.adapters.negotiate import NegotiationError, multi_gpu_decision
from fastmd.adapters.protocol import GraphBatch
from fastmd.engine.energy_derivative import EnergyDerivative
from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics
from fastmd.config import OPTION_REGISTRY
from fastmd.engine import memory_admission
from pair_potential_adapter import SystemTapPairAdapter
from test_isolated_atom_handling import make_model

ROOT = Path(__file__).resolve().parents[1]


class MigrationBoundaryTest(unittest.TestCase):
    def test_no_old_package_or_distributed_imports(self):
        for path in (ROOT / 'fastmd').rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                         [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                for name in names:
                    self.assertFalse(name == 'matris' or name.startswith(('matris.', 'fastmd.distributed')),
                                     f'{path}: {name}')
        self.assertFalse((ROOT / 'fastmd/distributed').exists())
        self.assertFalse(hasattr(GPUResidentMolecularDynamics, 'from_evaluator'))
        from fastmd.execution.md_protocol import RESTART_FORMAT
        self.assertEqual(RESTART_FORMAT, 'fastmd.distributed.md/1')

    def test_config_bindings_resolve_to_migrated_files(self):
        for option in OPTION_REGISTRY.values():
            for module, _ in option.bindings:
                self.assertTrue((ROOT / (module.replace('.', '/') + '.py')).is_file(), module)

    def test_external_model_names_dispatch_without_loading_weights(self):
        for name, module, loader in (
            ('mace_mp_0', 'mace', 'load_mace'),
            ('sevennet_0', 'sevennet', 'load_sevennet'),
            ('orb_v3', 'orb', 'load_orb'),
            ('chgnet_0_3_0', 'chgnet', 'load_chgnet'),
        ):
            with self.subTest(model=name):
                adapter = SystemTapPairAdapter()
                with patch(f'fastmd.adapters.{module}.{loader}', return_value=adapter) as load:
                    actual = load_adapter(name, device='cuda', optimization_profile='generic')
                self.assertIs(actual, adapter)
                load.assert_called_once_with(name, model_path=None, device='cuda', model_dtype=None,
                                             optimization_profile='generic')

    def test_rejects_multiple_devices_even_for_matris(self):
        for adapter in (load_adapter(make_model()), SystemTapPairAdapter()):
            self.assertEqual(multi_gpu_decision(adapter, 1).chosen, 'single_gpu')
            for size in (0, 2, 4):
                with self.assertRaisesRegex(NegotiationError, 'one device'):
                    multi_gpu_decision(adapter, size)
        self.assertTrue(all(key[0] == 'single' for key in memory_admission.CALIBRATION))

    def test_imports_without_old_or_optional_models(self):
        # This also catches accidental imports through an installed source checkout.
        code = '''
import importlib.abc, sys
blocked = {'matris', 'mace', 'sevenn', 'orb_models', 'chgnet', 'matgl'}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Block())
import fastmd
from fastmd import MLIPCalculator, GPUResidentMolecularDynamics
from fastmd.adapters import mace, sevennet, orb, chgnet
assert 'fastmd.models.matris.model' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                                text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class SimulationSmokeTest(unittest.TestCase):
    def test_checkpoint_load_calculator_and_ase_trajectory(self):
        torch.manual_seed(19)
        model = make_model()
        atoms = Atoms('H2', positions=[[1, 1, 1], [1.85, 1.07, 1.02]], cell=[8]*3, pbc=True)
        config = InferenceConfig(execution='eager', optimization_profile='generic', report=False)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'tiny.pth.tar'
            torch.save({'config': model.config, 'state_dict': model.state_dict()}, checkpoint)
            atoms.calc = MLIPCalculator(model_path=str(checkpoint), device='cpu', task='efs', config=config)
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            self.assertTrue(np.isfinite(energy))
            self.assertTrue(np.isfinite(atoms.get_stress()).all())
            # Check the adapter's total-energy convention and conservative force sign together.
            dx = 1e-3
            atoms.positions[0, 0] += dx
            plus = atoms.get_potential_energy()
            atoms.positions[0, 0] -= 2*dx
            minus = atoms.get_potential_energy()
            atoms.positions[0, 0] += dx
            self.assertAlmostEqual(forces[0, 0], -(plus-minus)/(2*dx), delta=3e-3)
            atoms.set_momenta([[0.01, 0, 0], [-0.01, 0, 0]])
            trajectory = Path(tmp) / 'md.traj'
            md = MolecularDynamics(atoms, model=atoms.calc.adapter, device="cpu", ensemble='nve', timestep=0.1,
                                   temperature=300, trajectory=str(trajectory), config=config)
            md.run(2)
            md.dyn.close()
            frames = read(trajectory, ':')
            self.assertEqual(len(frames), 3)
            self.assertTrue(np.isfinite(frames[-1].get_forces()).all())
            self.assertFalse(np.array_equal(frames[0].positions, frames[-1].positions))

    def test_external_energy_tap_forces_and_stress_without_matris(self):
        adapter = SystemTapPairAdapter()
        self.assertIs(load_adapter(adapter), adapter)
        positions = torch.tensor([[1., 1., 1.], [2.2, 1., 1.]], dtype=torch.float64)
        cell = torch.eye(3, dtype=torch.float64)*10
        batch = GraphBatch(positions=positions, frac_coords=positions/10, cell=cell,
                           species=torch.tensor([1, 8]), n_real=2,
                           edge_index=torch.tensor([[0, 1], [1, 0]]),
                           edge_image=torch.zeros(2, 3, dtype=torch.float64),
                           n_edges=torch.tensor([2]), cutoff=adapter.module.cutoff, padding=None)
        result = EnergyDerivative(adapter)(batch, adapter.prepare(batch), task='efs')
        r, rc, power = 1.2, adapter.module.cutoff, adapter.module.power
        strength = float((adapter.module.species_strength[1]*adapter.module.species_strength[8]).detach())
        expected_energy = strength*(1-(r/rc)**2)**power
        derivative = -2*strength*power*r/rc**2*(1-(r/rc)**2)**(power-1)
        self.assertAlmostEqual(result['e'].item(), expected_energy, delta=1e-10)
        torch.testing.assert_close(result['f'][0], torch.tensor([[derivative, 0, 0], [-derivative, 0, 0]],
                                                               dtype=torch.float64), rtol=1e-6, atol=1e-10)
        expected_stress = torch.zeros(3, 3, dtype=torch.float64)
        expected_stress[0, 0] = derivative*r/1000
        torch.testing.assert_close(result['s'][0], expected_stress, rtol=1e-6, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
