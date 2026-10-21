from __future__ import annotations

""" Modified from CHGNet: https://github.com/CederGroupHub/chgnet """

import inspect
import sys
import io
import contextlib
from ase import Atoms, units
from ase.optimize.optimize import Optimizer
import ase.filters as filter_classes
from ase.filters import Filter
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure

from fastmd.config import InferenceConfig
from fastmd.engine.entry_config import reject_removed_arguments
from fastmd.interfaces.calculator import OPTIMIZERS
from fastmd.interfaces.calculator import MLIPCalculator
from fastmd.interfaces.calculator import TrajectoryObserver
from fastmd.interfaces.calculator import CrystalFeasObserver

class StructOptimizer:
    """This class is used for Sturcture Optimization."""

    def __init__(
        self,
        model_path: str = None,
        model: str = "matris_10m_oam",
        task: str = "efs",
        optimizer: str = "FIRE",
        device: str = "cpu",
        enable_compile: bool = True,
        config: InferenceConfig | None = None,
    ) -> None:
        """
        Args:
            model_path (str): Local checkpoint path. If set, this checkpoint is loaded.
            model (MatRIS): Instance of a MatRIS model. If set to None, the default MatRIS is loaded.
            task (str): The prediction task. Can be 'e', 'em', 'ef', 'efs', 'efsm'.
            optimizer (Optimizer,str): choose optimizer from ASE.
            device (str): The device to be used for predictions,
            enable_compile (bool): Whether to compile interaction blocks.
            config: Unified inference settings forwarded to MLIPCalculator.
                Defaults to eager execution and checkpointing off.
        """
        
        if optimizer in OPTIMIZERS:
            optimizer = OPTIMIZERS[optimizer]
        else:
            raise ValueError(
                f"Optimizer {optimizer} not found. Select from {list(OPTIMIZERS)}"
            )
        
        self.optimizer: Optimizer = optimizer
        
        self.calculator = MLIPCalculator(
            model_path=model_path,
            model=model,
            task=task,
            device=device,
            enable_compile=enable_compile,
            config=config,
        )

    def config_summary(self) -> dict:
        """Return the relaxation calculator's effective inference settings."""
        return self.calculator.config_summary()

    def close(self) -> None:
        """Close the calculator this optimizer made (``MLIPCalculator.close``: its graphs, and the release
        of its model adapter); context exit closes too."""
        self.calculator.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
    
    def relax(
        self,
        atoms: Structure | Atoms,
        fmax: float = 0.05,
        steps: int = 500,
        relax_cell: bool = True,
        ase_filter: str = "FrechetCellFilter",
        save_path: str = None,
        loginterval: int = 1,
        crystal_feas_save_path: str = None,
        verbose: bool = True,
        assign_magmoms: bool = True,
        **kwargs,
    ) -> dict[str, Structure | TrajectoryObserver]:
        """
        Args:
            atoms (Structure | Atoms): A Structure or Atoms object to relax.
            fmax (float): The maximum force tolerance for relaxation.
            steps (int): The maximum number of steps for relaxation.
            relax_cell (bool): Whether to relax the cell as well.
            ase_filter (str): The filter to apply to the atoms object for relaxation. 
            save_path (str): The path to save the trajectory.
            loginterval (int): Interval for logging trajectory and crystal feas.
            crystal_feas_save_path (str): Path to save crystal feature vectors
                which are logged at a loginterval rage
            verbose (bool): Whether to print the output of the ASE optimizer.
            assign_magmoms (bool): Whether to assign magnetic moments to the final
                structure.
            **kwargs: Additional parameters for the optimizer.
        """

        reject_removed_arguments(kwargs)
        valid_filter_names = [
            name
            for name, cls in inspect.getmembers(filter_classes, inspect.isclass)
            if issubclass(cls, Filter)
        ]

        if isinstance(ase_filter, str):
            if ase_filter in valid_filter_names:
                ase_filter = getattr(filter_classes, ase_filter)
            else:
                raise ValueError(
                    f"Invalid {ase_filter=}, must be one of {valid_filter_names}. "
                )

        if isinstance(atoms, Structure):
            atoms = AseAtomsAdaptor().get_atoms(atoms)

        atoms.calc = self.calculator

        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)

            if crystal_feas_save_path:
                cry_obs = CrystalFeasObserver(atoms)

            if relax_cell:
                atoms = ase_filter(atoms)
            optimizer: Optimizer = self.optimizer(atoms, **kwargs)
            optimizer.attach(obs, interval=loginterval)

            if crystal_feas_save_path:
                optimizer.attach(cry_obs, interval=loginterval)

            optimizer.run(fmax=fmax, steps=steps)
            obs()

        if save_path is not None:
            obs.save(save_path)

        if crystal_feas_save_path:
            cry_obs.save(crystal_feas_save_path)

        if isinstance(atoms, Filter):
            atoms = atoms.atoms
        struct = AseAtomsAdaptor.get_structure(atoms)
        
        if assign_magmoms:
            if atoms.get_magnetic_moments() is not None:
                for key in struct.site_properties:
                    struct.remove_site_property(property_name=key)
                struct.add_site_property(
                    "magmom", [float(magmom) for magmom in atoms.get_magnetic_moments()]
                )
        
        return {"final_structure": struct, "trajectory": obs}
