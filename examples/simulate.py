"""Run single-device MD using any migrated MLIP adapter.

python examples/simulate.py --structure input.extxyz --model chgnet_0_3_0 --steps 100
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structure', required=True)
    parser.add_argument('--model', default='matris_10m_oam')
    parser.add_argument('--model-path')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--driver', choices=('gpu', 'ase'), default='gpu')
    parser.add_argument('--execution', choices=('auto', 'eager', 'model_graph', 'whole_step'), default='auto')
    parser.add_argument('--profile', choices=('topology', 'generic'), default='topology')
    parser.add_argument('--ensemble', choices=('nve', 'nvt', 'nvt_nhc'), default='nvt')
    parser.add_argument('--temperature', type=float, default=300)
    parser.add_argument('--timestep', type=float, default=0.5, help='fs')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--trajectory', default='trajectory.traj')
    parser.add_argument('--logfile', default='md.log')
    args = parser.parse_args()
    if args.steps < 0 or args.timestep <= 0 or args.temperature <= 0:
        parser.error('steps must be nonnegative; timestep and temperature must be positive')
    if args.driver == 'ase' and (args.execution == 'whole_step' or args.ensemble == 'nvt_nhc'):
        parser.error('whole_step and nvt_nhc in this example require --driver gpu')

    from ase.io import read
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from fastmd import InferenceConfig, GPUResidentMolecularDynamics, MolecularDynamics

    atoms = read(args.structure)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    config = InferenceConfig(execution=args.execution, optimization_profile=args.profile)
    options = dict(atoms=atoms, model=args.model, model_path=args.model_path, device=args.device,
                   config=config, ensemble=args.ensemble, temperature=args.temperature,
                   timestep=args.timestep, trajectory=args.trajectory, logfile=args.logfile)
    if args.driver == 'gpu':
        with GPUResidentMolecularDynamics(**options) as md:
            md.run(args.steps)
            print(md.thermodynamics())
    else:
        md = MolecularDynamics(**options)
        md.run(args.steps)
        print('Potential energy (eV):', md.atoms.get_potential_energy())


if __name__ == '__main__':
    main()
