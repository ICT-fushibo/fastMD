# FastMD

FastMD is a single-device simulation framework for machine-learned interatomic potentials (MLIPs). It provides a common interface for ASE calculators, structure relaxation, ASE molecular dynamics, and GPU-resident molecular dynamics, with `eager`, `model_graph`, and `whole_step` execution modes.

## Supported models

Checkpoint names link to official weight files or model pages.

| Model | Name or entry point | Model implementation |
| --- | --- | --- |
| MatRIS | [matris_10m_oam](https://ndownloader.figshare.com/files/59142728), [matris_10m_mp](https://ndownloader.figshare.com/files/59143058), or `model_path=...` | Bundled single-device implementation; weights loaded separately |
| MACE | [mace_mp_0](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model), [mace_mpa_0](https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model), `mace` + `model_path`, or a `.model` path | Optional `mace-torch` package |
| SevenNet | [sevennet_0](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_0__11Jul2024/checkpoint_sevennet_0.pth), [7net-0](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_0__11Jul2024/checkpoint_sevennet_0.pth), or `sevennet` + `model_path` | Optional `sevenn` package |
| ORB | [orb_v3](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-inf-mpa-20250404.ckpt), with optional `model_path` | Optional `orb-models` package |
| CHGNet | [chgnet_0_3_0](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar) | Pretrained weights supplied by the optional `chgnet` package |
| eSEN | [esen](https://huggingface.co/facebook/OMAT24#model-checkpoints) + `model_path` | Legacy `fairchem-core==1.10.0`; isolated environment and local weights required |
| Prophet-OAME-MBD | [Prophet-OAME-MBD](https://huggingface.co/kairosmaterial/prophet/blob/e22a79d16726484b7b635221cfe2b84a7bdf8a33/prophet-oame-mbd.pt) + `model_path` | Official `prophet-mlip`; PyTorch energy interface with eager execution |

The experimental M3GNet topology/sparse-operator integration is retained in `integrations/`; it is not yet a complete adapter for the unified MD interface. The TorchSim wrapper retains its original capability checks and does not support every MLIP.

## S01 benchmark results

The tables below reproduce the completed S01 results in the development workspace's `records2610/README.md`, including its published ORB operator retests, as reviewed on **October 21, 2026**. Detailed result files and provenance snapshots remain in the local benchmark archive; the published comparison is reproduced here.

These are historical S01 measurements from the recorded source snapshots. They are not a new run of every checkpoint against the current branch. The table contains **41 checkpoints across five model families**, with a complete official/FastMD pair for each checkpoint.

| Setting | Value |
| --- | --- |
| System | S01: `CsSnI3_500K_Ivor_VASP`, 135 atoms |
| Hardware | NVIDIA H100 80 GB HBM3; one GPU per trajectory |
| Ensemble | NVT Nose–Hoover chain at 500 K |
| Integration | 0.25 fs × 80,000 steps = 20 ps |
| Thermostat | `tdamp=25 fs`, `tchain=3`, `tloop=1` |
| Recording | One frame every 10 steps; 8,001 frames including the initial state |
| Precision | FP32 IEEE |
| Initial state | Shared coordinates, cell, and momenta; seed 0; no center-of-mass momentum removal |
| Timing | GPU-synchronized wall time for the complete MD run, including trajectory output; model loading, compilation, capture, and post-run validation excluded |

**Speedup = official full-run time / FastMD full-run time.** The displayed step times are the complete MD wall times divided by 80,000. Only complete runs enter these tables; short-run timings are excluded. FastMD's execution mode depends on the model and its capabilities. ORB direct/v2 and capped-neighbor retests use a compiled backbone and whole-step CUDA Graphs; SevenNet checkpoints that cannot meet exact padding requirements use eager execution.

<!-- s01-results:begin -->

| Model family | Completed checkpoint pairs | Recorded speedup range |
| --- | ---: | ---: |
| MatRIS | 2 | 5.13–5.52× |
| MACE | 15 | 12.88–24.73× |
| SevenNet | 8 | 14.87–24.47× |
| ORB | 13 | 3.23–7.19× |
| CHGNet | 3 | 18.45–21.05× |

### MatRIS

| Checkpoint | Official (ms/step) | FastMD (ms/step) | Speedup |
| --- | ---: | ---: | ---: |
| [matris_10m_oam](https://ndownloader.figshare.com/files/59142728) | 154.5975 | 28.0180 | 5.5178× |
| [matris_10m_mp](https://ndownloader.figshare.com/files/59143058) | 164.6329 | 32.0987 | 5.1290× |

### MACE

| Checkpoint | Official (ms/step) | FastMD (ms/step) | Speedup |
| --- | ---: | ---: | ---: |
| [mace-small](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model) | 31.8842 | 1.5295 | 20.8460× |
| [mace-medium](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model) | 43.0477 | 1.7405 | 24.7327× |
| [mace-large](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/MACE_MPtrj_2022.9.model) | 40.2385 | 2.1759 | 18.4924× |
| [mace-small-0b](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_small.model) | 31.5469 | 2.0938 | 15.0665× |
| [mace-medium-0b](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_medium.model) | 39.0663 | 2.3195 | 16.8429× |
| [mace-small-0b2](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-small-density-agnesi-stress.model) | 28.2565 | 2.1946 | 12.8752× |
| [mace-medium-0b2](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-medium-density-agnesi-stress.model) | 44.8428 | 2.5230 | 17.7738× |
| [mace-large-0b2](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-large-density-agnesi-stress.model) | 38.3883 | 2.8898 | 13.2842× |
| [mace-medium-0b3](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model) | 52.3621 | 2.5283 | 20.7101× |
| [mace-medium-mpa-0](https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model) | 43.0448 | 2.3935 | 17.9840× |
| [mace-small-omat-0](https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-small.model) | 41.8795 | 2.1667 | 19.3284× |
| [mace-medium-omat-0](https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model) | 41.1945 | 2.4880 | 16.5574× |
| [mace-mace-matpes-pbe-0](https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model) | 51.4085 | 2.5274 | 20.3405× |
| [mace-mace-matpes-r2scan-0](https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model) | 36.5004 | 2.4031 | 15.1891× |
| [mace-large-2024-01-07](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2024-01-07-mace-128-L2_epoch-199.model) | 36.9483 | 2.1359 | 17.2991× |

### SevenNet

| Checkpoint | Official (ms/step) | FastMD (ms/step) | Speedup |
| --- | ---: | ---: | ---: |
| [7net-0](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_0__11Jul2024/checkpoint_sevennet_0.pth) | 53.6565 | 2.5952 | 20.6754× |
| [7net-0_22may2024](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_0__22May2024/checkpoint_sevennet_0.pth) | 62.0417 | 2.5770 | 24.0756× |
| [7net-l3i5](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_l3i5/checkpoint_l3i5.pth) | 80.1659 | 3.2757 | 24.4730× |
| [7net-omat](https://github.com/MDIL-SNU/SevenNet/releases/download/v0.11.0.cp/checkpoint_sevennet_omat.pth) | 59.0921 | 3.8197 | 15.4705× |
| [7net-nano-4.5](https://github.com/MDIL-SNU/SevenNet/releases/download/v0.13.1.cp/checkpoint_7net_nano_4.5.pth) | 34.0357 | 1.4519 | 23.4430× |
| [7net-nano-5.0](https://github.com/MDIL-SNU/SevenNet/releases/download/v0.13.1.cp/checkpoint_7net_nano_5.0.pth) | 20.3308 | 1.3669 | 14.8732× |
| [7net-nano-5.5](https://github.com/MDIL-SNU/SevenNet/releases/download/v0.13.1.cp/checkpoint_7net_nano_5.5.pth) | 32.0208 | 1.4654 | 21.8514× |
| [7net-nano-6.0](https://github.com/MDIL-SNU/SevenNet/releases/download/v0.13.1.cp/checkpoint_7net_nano_6.0.pth) | 23.2114 | 1.3616 | 17.0477× |

### ORB

| Checkpoint | Official (ms/step) | FastMD (ms/step) | Speedup |
| --- | ---: | ---: | ---: |
| [orb-v3-conservative-20-omat](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-20-omat-20250404.ckpt) | 34.3849 | 5.2446 | 6.5562× |
| [orb-v3-conservative-inf-omat](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-inf-omat-20250404.ckpt) | 36.7809 | 5.1188 | 7.1855× |
| [orb-v3-direct-20-omat](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-direct-20-omat-20250404.ckpt) | 8.8192 | 2.5000 | 3.5277× |
| [orb-v3-direct-inf-omat](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-direct-inf-omat-20250404.ckpt) | 9.6175 | 2.7153 | 3.5420× |
| [orb-v3-conservative-20-mpa](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-20-mpa-20250404.ckpt) | 28.9682 | 4.7892 | 6.0487× |
| [orb-v3-conservative-inf-mpa](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-inf-mpa-20250404.ckpt) | 33.8396 | 5.0916 | 6.6461× |
| [orb-v3-direct-20-mpa](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-direct-20-mpa-20250404.ckpt) | 9.0028 | 2.7347 | 3.2921× |
| [orb-v3-direct-inf-mpa](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-direct-inf-mpa-20250404.ckpt) | 8.9940 | 2.7543 | 3.2654× |
| [orb-v2](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v2-20241011.ckpt) | 11.1864 | 3.4686 | 3.2251× |
| [orb-d3-v2](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-d3-v2-20241011.ckpt) | 11.5781 | 3.4350 | 3.3706× |
| [orb-d3-sm-v2](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-d3-sm-v2-20241011.ckpt) | 8.6669 | 2.5232 | 3.4349× |
| [orb-d3-xs-v2](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-d3-xs-v2-20241011.ckpt) | 6.7959 | 1.7014 | 3.9943× |
| [orb-mptraj-only-v2](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-mptraj-only-v2-20241014.ckpt) | 10.9139 | 3.2807 | 3.3267× |

### CHGNet

| Checkpoint | Official (ms/step) | FastMD (ms/step) | Speedup |
| --- | ---: | ---: | ---: |
| [chgnet-0.3.0](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar) | 53.3781 | 2.7588 | 19.3484× |
| [chgnet-0.2.0](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/pretrained/0.2.0/chgnet_0.2.0_e30f77s348m32.pth.tar) | 52.0884 | 2.4750 | 21.0455× |
| [chgnet-r2scan](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/pretrained/r2scan/chgnet_r2scan_transfer_learning_e15f36s161m23.pth.tar) | 51.5925 | 2.7956 | 18.4548× |

<!-- s01-results:end -->

The S01 campaign excluded [mace-mh-0](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-0.model), [mace-mh-1](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model), [7net-mf-0](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_MF_0/checkpoint_sevennet_mf_0.pth), [7net-mf-ompa](https://github.com/MDIL-SNU/SevenNet/releases/download/v0.11.0.cp/checkpoint_sevennet_mf_ompa.pth), [7net-omni](https://github.com/MDIL-SNU/SevenNet/releases/download/v0.12.0.cp/checkpoint_sevennet_omni.pth), [7net-omni-i8](https://github.com/MDIL-SNU/SevenNet/releases/download/v0.12.1.cp/checkpoint_sevennet_omni_i8.pth), and [7net-omni-i12](https://github.com/MDIL-SNU/SevenNet/releases/download/v0.12.1.cp/checkpoint_sevennet_omni_i12.pth); MACE molecular models were outside its scope. eSEN was an additional campaign model and is not part of the 41-row source table.

The published ORB operator-retest replacements are [orb-v3-direct-20-omat](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-direct-20-omat-20250404.ckpt), [orb-v3-conservative-20-mpa](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-20-mpa-20250404.ckpt), and [orb-mptraj-only-v2](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-mptraj-only-v2-20241014.ckpt). Other rows retain their previously validated complete runs. The locally archived evidence includes CPU initial energy/force/stress checks for all 41 checkpoints, separate passing GPU initial-parity files for 40, and 17-frame trajectory checks for the 11 ORB capture/operator replacements. MatRIS OAM has no separate archived GPU initial-parity file. The full table does not imply 17-frame E/F/S validation for every model.

At the October 21 snapshot, the separate Prophet S01 campaign had completed the official trajectory (**276.6853 ms/step**) while FastMD was still running, so no Prophet speedup is reported. GRACE had only completed its official smoke test and has no FastMD adapter in this branch; it is excluded from the comparison. These incomplete runs do not contribute to the completed-checkpoint counts above. A historical S01 result does not establish accuracy on other structures or newer source snapshots; current S03 limitations are listed under [Validation](#validation).

## Installation

Use Python 3.10 or newer and run these commands from the repository root. Base and optional dependencies are managed in `pyproject.toml`:

```bash
python -m pip install -e .
# GPU neighbor-graph construction:
python -m pip install -e '.[gpu-graph]'
# CHGNet with GPU graph construction:
python -m pip install -e '.[gpu-graph,chgnet]'
# Install other model backends as needed:
python -m pip install -e '.[gpu-graph,mace]'
# Experimental M3GNet integration:
python -m pip install -e '.[m3gnet]'
```

The base installation requires Torch ≥ 2.6. The `gpu-graph` extra requires Torch ≥ 2.8, `nvalchemi-toolkit-ops[torch]>=0.4.1,<0.5`, and `warp-lang>=1.13.0`. Installing this extra can upgrade an older Torch installation; prepare a Torch build matching your CUDA environment first. These dependencies provide the GPU neighbor-list backend. The former `graph/setup.py` and `fastmd_op` build are no longer required. Use `-e .` and `-e '.[m3gnet]'` in place of the old `requirements.txt` and `requirements-m3gnet.txt` installation paths.

The MACE and SevenNet `topology` profiles use cuEquivariance. The upstream CUDA 13 dependency set is available through `python -m pip install -e '.[gpu-graph,mace,cueq-cu13]'`; substitute `sevennet` for `mace` as needed. Use `optimization_profile="generic"` without that acceleration environment. MACE and SevenNet require different e3nn versions and should use separate environments.

`requirements-{mace,sevennet,orb,chgnet}.txt` preserve upstream test-environment versions and hashes. They do not replace the base installation, contain Python/platform-specific wheel hashes, and are not complete dependency lockfiles. Use the extras above for normal installation. The upstream CI scripts mentioned in those records are not included here.

The migrated third-party adapter paths currently require CUDA neighbor lists and three-dimensional periodic cells. MatRIS also retains a CPU calculator path. Model packages are loaded lazily. Loading a model by name may download weights; pass a local checkpoint to avoid the corresponding download.

### eSEN and Prophet environments

Legacy eSEN uses an isolated environment because fairchem 1.10.0 conflicts with the base package's dependency constraints. The recorded environment uses Python 3.11, fairchem-core 1.10.0, Torch 2.8.0+cu128, NumPy 1.26.4, SciPy 1.14.1, e3nn 0.5.9, ASE 3.26.0, nvalchemi-toolkit-ops 0.2.0, and warp-lang 1.10.0, with PyG extensions matching Torch/CUDA. Torch 2.8 is an explicit override of fairchem's declared Torch 2.4 requirement. There is no `esen` extra: use the prepared environment and load this checkout through `PYTHONPATH`, for example `PYTHONPATH="$PWD" python your_script.py` from the repository root. Supply an [eSEN-30M-OAM](https://huggingface.co/facebook/OMAT24#model-checkpoints) checkpoint explicitly; fairchem 2.x/UMA and direct-force heads are not supported by this adapter.

Prophet requires Python ≥ 3.11 and a separate environment with e3nn ≥ 0.6 and ASE ≥ 3.29. Install it with `python -m pip install -e '.[gpu-graph,prophet]'`; optional official CUDA tensor-product kernels are available through `openequivariance>=0.6` with a compatible CUDA toolchain. Supply the [Prophet-OAME-MBD](https://huggingface.co/kairosmaterial/prophet/blob/e22a79d16726484b7b635221cfe2b84a7bdf8a33/prophet-oame-mbd.pt) checkpoint through `model_path`. This adapter uses eager execution because the official network has dynamic masks. Controlled comparisons must use the same kernel backend for the official calculator and FastMD.

## Unified calculator

```python
from ase.build import bulk
from fastmd import MLIPCalculator, InferenceConfig

atoms = bulk("Si", "diamond", a=5.43, cubic=True)
atoms.calc = MLIPCalculator(
    model="chgnet_0_3_0", device="cuda", task="efs",
    config=InferenceConfig(execution="eager", optimization_profile="generic"),
)
print(atoms.get_potential_energy())  # eV, total energy
print(atoms.get_forces())           # eV/Å
print(atoms.get_stress())           # eV/Å³, ASE Voigt order
```

For a CPU example, use `model="matris_10m_oam"` and `device="cpu"`. The adapters and shared derivative layer handle model energy normalization, gradients, and stress units while preserving ASE conventions at the interface.

## Molecular dynamics

```bash
python examples/simulate.py --structure input.extxyz \
  --model chgnet_0_3_0 --driver gpu --device cuda \
  --ensemble nvt_nhc --temperature 300 --timestep 0.5 \
  --execution auto --steps 100
```

You can also instantiate `GPUResidentMolecularDynamics` directly. Use it as a context manager to finish writing logs and trajectories on exit. `MolecularDynamics` provides ASE-driven MD, and `StructOptimizer` provides structure relaxation. `execution="auto"` selects a path using model capabilities, available memory, and applicable constraints; it does not guarantee whole-step capture for every model.

```bash
fastmd-explain --structure input.extxyz --model chgnet_0_3_0 --device cuda --json
```

## Structure relaxation

The relaxation API ports the task-level behavior of [`hsy_relax`](https://github.com/littleyu192/fastMD/tree/hsy_relax)
onto the current `MLIPCalculator`. Relaxation and MD share model adapters, energy derivatives,
operator optimizations, neighbor construction, and `InferenceConfig`. There is no separate
relaxation model backend. The older branch's `FastMDCalculator` and `CUDAGraphConfig` are
replaced by `MLIPCalculator` and `InferenceConfig` here.

```python
from ase.io import read, write
from fastmd import InferenceConfig, RelaxationConfig, make_relaxation_calculator, relax

atoms = read("input.extxyz")
settings = RelaxationConfig(relax_cell=True, optimizer="FIRE", fmax=0.05, steps=500)
with make_relaxation_calculator(
    model="chgnet_0_3_0", device="cuda", relax_cell=settings.relax_cell,
    config=InferenceConfig(execution="auto", optimization_profile="topology"),
) as calculator:
    result = relax(atoms, calculator, settings)
    print(result.converged, result.steps, result.final_energy, result.max_filter_force)
    print(calculator.config_summary()["effective"])
    # Reuse this same calculator for further structures; each gets a fresh optimizer.
write("relaxed.extxyz", result.atoms)
```

Use `relax_cell=False` in both settings and the calculator factory for fixed-cell relaxation.
The factory fixes the prediction task to `ef` for a fixed cell and `efs` for a variable cell,
so ASE's repeated property requests reuse each geometry's evaluated results. The optimizer
choices are `FIRE`, `BFGS`, `LBFGS`, `BFGSLineSearch`, `LBFGSLineSearch`, and `MDMin`.
Pass optimizer parameters through `optimizer_kwargs`, and `FrechetCellFilter` parameters
such as `mask`, `hydrostatic_strain`, or `scalar_pressure` through `filter_kwargs`.
Pressure uses ASE units (eV/Å³).

`relax` copies the input atoms, retains constraints, and returns an independent final geometry
with a `SinglePointCalculator`. Reusing or closing the inference calculator cannot invalidate
previous results. It never closes a calculator supplied by the caller. `steps=0` evaluates
the initial structure. `max_force` is the maximum constrained atomic force; for variable-cell
relaxation, convergence uses `max_filter_force`, including scaled cell forces, rather than
atomic forces alone. `fmax` is therefore not a separate stress tolerance. Non-finite results
raise an error. Timing includes initial evaluation and graph capture, but excludes model loading.

The existing `StructOptimizer` interface also accepts `calculator=calculator`, keeps its
`final_structure` / `trajectory` return keys, and reports convergence and optimizer steps.
For an active ASE `MolecularDynamics` instance, `relax(md.atoms, md.calculator, settings)`
reuses its calculator and cached inference state; a cell relaxation requires its task to
include stress. Use a shared calculator sequentially. To transfer a model from GPU-resident MD,
pass `model=md.adapter` to the factory: the weights and adapter are reused, while the ASE
calculator manages its own execution state.

| Path | Optimization reuse | Requirements |
| --- | --- | --- |
| `relax` / `StructOptimizer` | Common calculator, operator kernels, compilation where supported, `eager` or `model_graph` | Model's calculator support; stress and a full periodic cell for variable-cell relaxation |
| `MultiSystemRelaxation` | Existing multi-system engine, slot batching, GPU FIRE and whole-step CUDA Graphs | CUDA, union/per-edge-shift adapter capabilities, full periodic cells, no ASE constraints |

`execution="auto"` uses the calculator's existing capability and memory decisions. Explicit
`model_graph` requests must satisfy those checks. ASE optimization runs on the host, so
`whole_step` is not an execution mode for the functional API. Variable cells continue through
the shared calculator's cell/topology invalidation and graph-capacity logic. Graph capture
is not guaranteed to improve short relaxations; inspect the effective configuration and
measure the workload. The third-party CUDA/PBC requirements in [Installation](#installation)
also apply to relaxation.

The CLI can process one structure or a multi-frame file with one loaded model:

```bash
fastmd-relax --structure input.extxyz --model chgnet_0_3_0 --device cuda \
  --execution auto --fmax 0.05 --steps 500 --output relaxed.extxyz

# Fixed cell; all frames, fresh optimizers, shared model and inference caches:
python -m fastmd.relax_cli --structure inputs.extxyz --index ':' --fixed-cell \
  --model mace --model-path /path/to/model.model --device cuda \
  --trajectory-dir relax-trajectories --output relaxed.extxyz --summary relaxation.json
```

`python examples/relax.py` exposes the same options. Final structures are written as
`.extxyz` or `.traj`, and the JSON summary includes per-structure energies, convergence,
step counts, timing, and the final calculator configuration. Exit code 0 means every
structure converged; exit code 1 means at least one reached the step limit, with its results
still written. This CLI processes frames sequentially. For supported GPU batched workloads:

```python
from ase.io import read
from fastmd import InferenceConfig, MultiSystemRelaxation

with MultiSystemRelaxation(
    model="mace", model_path="/path/to/model.model",
    structures=read("inputs.extxyz", ":"), device="cuda",
    cell_filter="frechet", fmax=0.05, max_steps=500,
    config=InferenceConfig(execution="whole_step", optimization_profile="topology"),
) as runner:
    results = runner.run()
```

Use `cell_filter=None` for fixed cells. Batched execution retains its existing capability
checks; it does not accept every calculator backend. The old branch's Discovery dataset
workflow is separate from the relaxation API and is not included in this port.

## Layout and extension points

```text
fastmd/
  models/            Model adapters, capabilities, patches, and the MatRIS implementation
  engine/            Configuration, model protocols, derivatives, capacity, and caches
  execution/         Integrators, thermostats, CUDA Graphs, logging, and batched execution
  interfaces/        ASE calculators, MD, relaxation, and multi-system interfaces
  graph/             Neighbor lists, atom/line graphs, fixed-capacity layouts, and topology
  ops/               Shared PyTorch, Triton, and CUDA operators
  integrations/      Experimental M3GNet topology integration
  adapters/          Compatibility forwarding for legacy adapter imports
  applications/      Compatibility forwarding for legacy application imports
  config.py          Configuration compatibility entry point
  precision.py       Precision configuration and records
```

To add an MLIP, implement `ModelAdapter` (available through `adapters/protocol.py`) and expose an `EnergyTap`, `SystemEnergyTap`, or `ModelOwnedDerivative` interface. Pass the adapter as `model=` to reuse the simulation runtime. Declare fixed-shape capture, padding, and cutoff-continuity capabilities accurately so the runtime can select a supported execution path.

Historical MatRIS-specific `MATRIS_*` operator settings are retained. The common configuration entry point is `InferenceConfig`. MatRIS weights use `~/.cache/matris`, and its restart format remains compatible with the source implementation.

### ORB direct/v2 models and neighbor limits

Load local weights using the upstream `ORB_PRETRAINED_MODELS`, wrap the model with `fastmd.adapters.orb_graph.ORBGraphAdapter(module)`, and pass the adapter as `model=` to the calculator or GPU MD interface. The adapter targets the validated `orb-models==0.5.4` architecture with a 6 Å cutoff and sigmoid message gating, for a single nonsingular periodic cell.

Every evaluation retains the nearest K neighbors per atom by distance, with K taken from `system_config.max_num_neighbors`. Padding edges and edges beyond K contribute zero. Direct/v2 models retain their direct force and stress heads; conservative models use the engine's energy derivatives. Both paths support backbone compilation and whole-step CUDA Graphs without changing weights, neighbor limits, or the default FP32 precision.

ORB and legacy eSEN also provide explicitly enabled candidate operator rewrites, including compact Top-K selection, message aggregation, and equivariant rotation fusion. These are experimental opt-in paths whose numerical and performance validation depends on the model and structure.

## Validation

```bash
PYTHONPATH=.:tests PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  python -B -m unittest discover -s tests
```

The suite covers migration boundaries, CPU model/ASE regressions without weight downloads, third-party adapters, and GPU graph/trajectory behavior. Tests skip when GPUs, optional dependencies, or external assets are unavailable; a skip is not a passing GPU validation. Full benchmark runners and detailed development records are maintained locally.

Pre-publication checks on October 21, 2026: `pytest tests -q --disable-warnings` completed with **153 passed and 187 skipped** in the CPU environment. Wheel construction and source/license-content checks passed. The separate H100 S03 long-trajectory campaign is ongoing. In that system, [orb-v3-conservative-20-mpa](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-20-mpa-20250404.ckpt) and [orb-v3-conservative-20-omat](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-20-omat-20250404.ckpt) exceeded the initial force-parity tolerance of 0.001 eV/Å; those results are excluded from speedup statistics.
