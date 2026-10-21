"""
    This code is referenced from: https://github.com/CederGroupHub/chgnet/blob/main/chgnet/graph/converter.py
    The original implementation can be found at the link above.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from fastmd.graph.gpu_graph_builder import atoms_to_graph_gpu
from fastmd.graph.gpu_graph_builder import op_available
from fastmd.graph.neighbor_list_backend import UNAVAILABLE_REASON
from fastmd.graph.matris_topology import matris_builder_evidence
from fastmd.graph.radiusgraph import RadiusGraph
from fastmd.models.matris.legacy_graph import Graph, Node

datatype = torch.float32

_CPU_BUILDER_EVIDENCE = matris_builder_evidence(
    "fastmd.cpu_radius_graph",
    atom_target_sorted=False,
    line_owner_sorted=False,
)


class GraphConverter(nn.Module):
    """Convert a pymatgen.core.Structure to a RadiusGraph"""

    def __init__(
        self,
        atom_graph_cutoff: float = 6,
        line_graph_cutoff: float = 4,
        verbose: bool = False,
    ) -> None:
        """Initialize the Graph Converter.
        
        Args:
            atom_graph_cutoff (float): cutoff radius in atom graph.
            line_graph_cutoff (float): bond length threshold in line graph.
            verbose (bool): whether to print the GraphConverter.
        """
        super().__init__()
        self.atom_graph_cutoff = atom_graph_cutoff
        self.line_graph_cutoff = (
            atom_graph_cutoff if line_graph_cutoff is None else line_graph_cutoff
        )
        self.create_graph = self._create_graph_legacy
        self.algorithm = 'gpu' if op_available else 'legacy'
        if not op_available:
            print(f"fast graph converter unavailable, using legacy: {UNAVAILABLE_REASON}")
        
        if verbose:
            print(self)

    def __repr__(self) -> str:
        """String representation of the GraphConverter."""
        atom_graph_cutoff = self.atom_graph_cutoff
        line_graph_cutoff = self.line_graph_cutoff
        algorithm = self.algorithm
        cls_name = type(self).__name__
        return f"{cls_name}({algorithm=}, {atom_graph_cutoff=}, {line_graph_cutoff=})"

    def forward(
        self,
        structure: Structure,
        graph_id=None,
        mp_id=None,
        atoms=None,
        check_isolated_atoms: bool = True,
    ) -> RadiusGraph:
        """Convert a structure, return a RadiusGraph.

        Args:
            structure (pymatgen.core.Structure): structure to convert
            graph_id (str): an id to keep track of this crystal graph
                Default = None
            mp_id (str): Materials Project id of this structure
                Default = None
            check_isolated_atoms (bool): Reject zero-neighbor atoms when True.
                False retains every atom, including an entirely empty edge
                graph, for a model/application with an explicit isolation policy.
        
        """
        if op_available and torch.cuda.is_available():
            # Use ASE atoms directly when provided (avoids pymatgen round trip).
            ase_atoms = atoms if atoms is not None else AseAtomsAdaptor.get_atoms(structure)
            graph = atoms_to_graph_gpu(
                ase_atoms,
                atom_graph_cutoff=self.atom_graph_cutoff,
                line_graph_cutoff=self.line_graph_cutoff,
                check_isolated_atoms=check_isolated_atoms,
            )
            graph.graph_id = graph_id
            graph.mp_id = mp_id
            graph.composition = (
                structure.composition.formula if structure is not None
                else ase_atoms.get_chemical_formula()
            )
            return graph

        n_atoms = len(structure)
        atomic_number = torch.tensor( [site.specie.Z for site in structure], dtype=torch.int32 )
        
        atom_frac_coord = torch.tensor( structure.frac_coords, dtype=datatype )
        lattice = torch.tensor( structure.lattice.matrix, dtype=datatype )

        center_index, neighbor_index, image, distance = structure.get_neighbor_list(
            r=self.atom_graph_cutoff, sites=structure.sites, numerical_tol=1e-8
        )
        # Ceate atom graph
        graph = self.create_graph(
            n_atoms, center_index, neighbor_index, image, distance
        )
        atom_graph, directed2undirected = graph.adjacency_list()
        atom_graph = torch.tensor(atom_graph, dtype=torch.int32).reshape(-1, 2)
        directed2undirected = torch.tensor(directed2undirected, dtype=torch.int32)
        undirected2directed = graph.undirected2directed()
        undirected2directed = torch.tensor(undirected2directed, dtype=torch.int32)
        
        line_graph = []
        try:
            line_graph = graph.line_graph_adjacency_list(
                cutoff=self.line_graph_cutoff
            ) 
        except Exception as exc:
            structure.to(filename="error_graph.cif")

        line_graph = torch.tensor(line_graph, dtype=torch.int32).reshape(-1, 5)

        # Keep the same explicit strict/deferred policy on CPU and GPU.
        n_isolated_atoms = len({*range(n_atoms)} - {*center_index})
        if check_isolated_atoms and n_isolated_atoms:
            error = f"Error: Detected {n_isolated_atoms} isolated atom. Calculation stopped"
            raise ValueError(error) # or print(error)
        
        return RadiusGraph(
            atomic_number=atomic_number,
            atom_frac_coord=atom_frac_coord,
            atom_graph=atom_graph,
            neighbor_image=torch.tensor(image, dtype=datatype).reshape(-1, 3),
            directed2undirected=directed2undirected,
            undirected2directed=undirected2directed,
            line_graph=line_graph,
            lattice=lattice,
            graph_id=graph_id,
            mp_id=mp_id,
            composition=structure.composition.formula,
            atom_graph_cutoff=self.atom_graph_cutoff,
            line_graph_cutoff=self.line_graph_cutoff,
            isolated_atom_count=torch.tensor([n_isolated_atoms], dtype=torch.int64),
            topology_evidence=_CPU_BUILDER_EVIDENCE,
        )

    @staticmethod
    def _create_graph_legacy(
        n_atoms: int,
        center_index: np.ndarray,
        neighbor_index: np.ndarray,
        image: np.ndarray,
        distance: np.ndarray,
    ) -> Graph:
        """Given structure information, create a Graph structure to be used to
        create Crystal_Graph using pure python implementation.

        Args:
            n_atoms (int): the number of atoms in the structure
            center_index (np.ndarray): np array of indices of center atoms.
                [num_undirected_bonds]
            neighbor_index (np.ndarray): np array of indices of neighbor atoms.
                [num_undirected_bonds]
            image (np.ndarray): np array of images for each edge.
                [num_undirected_bonds, 3]
            distance (np.ndarray): np array of distances.
                [num_undirected_bonds]

        Return:
            Graph data structure used to create Crystal_Graph object
        """
        
        graph = Graph([Node(index=idx) for idx in range(n_atoms)])
        for ii, jj, img, dist in zip(center_index, neighbor_index, image, distance):
            graph.add_edge(center_index=ii, neighbor_index=jj, image=img, distance=dist)
      
        return graph

    def as_dict(self) -> dict[str, float]:
        """Save the args of the graph converter."""
        return {
            "atom_graph_cutoff": self.atom_graph_cutoff,
            "line_graph_cutoff": self.line_graph_cutoff,
            "algorithm": self.algorithm,
        }

    @classmethod
    def from_dict(cls, dict) -> GraphConverter:
        """Create converter from dictionary."""
        return GraphConverter(**dict)
