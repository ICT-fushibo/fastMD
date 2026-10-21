from typing import Literal, Union, Sequence

import torch
from torch import Tensor, nn
import os
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from functools import partial
import threading

from fastmd.graph import RadiusGraph, datatype
from fastmd.models.matris.converter import GraphConverter
from fastmd.models.matris.reference_energy import AtomRef
from fastmd.models.matris.processgraph import process_graphs
from fastmd.models.matris.feature_embed import ThreebodyFourierExpansion
from fastmd.models.matris.feature_embed import AtomTypeEmbedding
from fastmd.models.matris.feature_embed import EdgeBasisEmbedding
from fastmd.models.matris.feature_embed import ThreebodyEmbedding
from fastmd.models.matris.functions import MLP
from fastmd.models.matris.functions import GatedMLP
from fastmd.models.matris.functions import get_normalization
from fastmd.models.matris.functions import segment_count
from fastmd.models.matris.interaction_block import Interaction_Block
from fastmd.models.matris.readout import EnergyHead
from fastmd.models.matris.readout import MagmomHead
from fastmd.models.matris.readout import ForceStressHead
from fastmd.config import PLAIN_TORCH_OPTIONS
from fastmd.config import bind_options
from fastmd.config import plain_torch_config
from fastmd.engine.policy import set_fusion_scope
from fastmd.models.matris import functions as _functions, interaction_block as _interaction_block
from fastmd.engine import saved_recompute as _saved_recompute


class _PlainTorchDispatch:
    """Bind the plain torch dispatch, reference-counted process-wide.

    Also the recompute context of activation checkpoints taken in training,
    which recompute in later backward passes (e.g. the loss backward, outside
    MatRIS.forward) on autograd threads, possibly while the config scope owner
    waits: a short private lock, never the scope lock, and the first entry
    binds while the last exit restores, in whatever order threads leave.
    """

    _lock = threading.Lock()
    _depth = 0
    _saved = None

    def __enter__(self):
        cls = _PlainTorchDispatch
        with cls._lock:
            if cls._depth == 0:
                options = bind_options(PLAIN_TORCH_OPTIONS)
                options.__enter__()
                cls._saved = (
                    options,
                    set_fusion_scope(False),
                    _functions.set_fused_activations(False),
                )
            cls._depth += 1

    def __exit__(self, *exc):
        cls = _PlainTorchDispatch
        with cls._lock:
            cls._depth -= 1
            if cls._depth == 0:
                options, scope, activations = cls._saved
                cls._saved = None
                _functions.set_fused_activations(activations)
                set_fusion_scope(scope)
                options.__exit__(None, None, None)


@contextmanager
def plain_torch_scope():
    """Run every fused/custom kernel dispatch site on its plain torch path.

    MatRIS.forward enters this scope for is_training=True, the single training
    predicate: readout then differentiates forces/stress with create_graph.
    Fused kernels are inference-only; several have a non-differentiable backward.
    It binds all kernel-selecting options (fastmd.config.PLAIN_TORCH_OPTIONS)
    under the config scope lock, clears the fusion scope (fastmd.engine.policy) and
    the fused activations, and re-binds all of it around the recomputation of
    activation checkpoints taken in the scope. Like any unmanaged call, a
    training step (forward and backward) must not run concurrently with
    configured inference scopes. enable_compile stays inference-only: compiled
    interaction blocks keep their own kernels.
    """
    dispatch = _PlainTorchDispatch()
    with plain_torch_config().scope(), dispatch:
        checkpoint = _interaction_block.checkpoint
        _interaction_block.checkpoint = partial(checkpoint, context_fn=lambda: (nullcontext(), dispatch))
        try:
            yield
        finally:
            _interaction_block.checkpoint = checkpoint


class MatRIS(nn.Module):
    """ Init MatRIS Potential """
    
    # saved_tensor_recompute outside an entry point's configuration scope
    # ("off"/"narrow"/"auto"; the P>1 evaluator sets it, P=1 entry points use
    # their scope).
    saved_recompute = "off"

    def __init__(
        self,
        num_layers: int = 6,
        node_feat_dim: int = 128,
        edge_feat_dim: int = 128,
        three_body_feat_dim: int = 128,
        mlp_hidden_dims: Union[int, Sequence[int]] = (128, 128),
        dropout: float = 0.0,
        use_bias: bool = False, 
        distance_expansion: str = "Bessel", 
        three_body_expansion: str = "SH",
        num_radial: int = 7,
        num_angular: int = 7,
        max_l: int = 4,
        max_n: int = 4,
        envelope_exponent: int = 8,
        graph_conv_mlp: str = "GateMLP",
        activation_type: str = "silu",
        norm_type: str = "rms",
        pairwise_cutoff: float = 6,
        three_body_cutoff: float = 4,
        use_smoothed_for_delta_edge: bool = False,
        learnable_basis: bool = True,
        is_intensive: bool = True,
        is_conservation: bool = True,
        reference_energy: str | None = None,
        enable_compile: bool = False,
        enable_checkpoint: bool | None = False,
    ):
        """
        Args:
            num_layers (int): message passing layers.
            node_feat_dim (int): atom feature embedding dim.
            edge_feat_dim (int): edge(pairwise) feature embedding dim.
            three_body_feat_dim (int): angle(three body) feature embedding dim.
            mlp_hidden_dims (List or int): hidden dims of MLP. 
                Can be 'int' or 'list'.
            dropout (float): dropout rate in MLP.
            use_bias (bool): whether use bias in Interaction block.
            distance_expansion (str):  The function of pairwise basis. 
                Can be "Bessel" or "Gaussian".
            three_body_expansion (str): The function of three body basis. 
                Can be "Fourier(fourier)" or "Spherical Harmonics(sh)".
            num_radial (int): number of radial basis used in Bessel and Gaussian basis.
            num_angular (int): number of three_body basis used in Fourier basis.
            max_l (int): Maximum l value for Spherical Harmonics basis (SH).
            max_n (int): Maximum n value for Spherical Harmonics basis (SH).
            envelope_exponent (int): exponent of 'PolynomialEnvelope'.
            graph_conv_mlp (str): The type of MLP in mp layers. 
                Can be "MLP", "GatedMLP" and "MoE". 
                See fucntion.py for more informations.
            activation_type (str): activation function. 
                Can be "SiLU(silu)", "Sigmoid(sigmoid)", "ReLU(relu)"...
                See fucntion.py for more informations.
            norm_type (str): normalization function used in MLP.
                Can be "LayerNorm(layer)", "BatchNorm(batch)", "RMSNorm(rms)"...
                See fucntion.py for more informations.
            pairwise_cutoff (float): The cutoff of Atom graph.
            three_body_cutoff (float): The cutoff of Line graph.
            use_smoothed_for_delta_edge (bool): Whether to use the smoothed features for edge feature update.
            learnable_basis (bool): Whether the basis functions are learnable.
            is_intensive (bool): whether the model outputs energy per atom (True) or total energy (False).
            is_conservation (bool): whether use conservate force and stress.
            reference_energy (str): refernece energy of 'str'(eg. MPtrj, OMat..) dataset(Caculated by linear regression).
                more details can be found at reference_energy.py.
            enable_compile (bool): Whether to compile interaction blocks.
            enable_checkpoint (bool | None): False disables activation
                checkpointing for inference performance, True forces
                checkpointing on, and None uses the automatic threshold.
        """
        
        super().__init__()
        # model configs
        self.config = { k: v for k, v in locals().items() if k not in ["self", "__class__"] }

        self.is_intensive = is_intensive
        self.enable_compile = enable_compile
        self.enable_checkpoint = enable_checkpoint
        
        self.reference_energy = None
        if reference_energy is not None:
            self.reference_energy = AtomRef(
                reference_energy=reference_energy,
                is_intensive=is_intensive
            ) 
        
        # Define Graph Converter
        self.graph_converter = GraphConverter(
            atom_graph_cutoff=pairwise_cutoff,
            line_graph_cutoff=three_body_cutoff,
        )

        # ====== embedding layers ========
        self.atom_embedding = AtomTypeEmbedding(atom_feat_dim=node_feat_dim)
        self.edge_embedding = EdgeBasisEmbedding(
            pairwise_cutoff=pairwise_cutoff,
            three_body_cutoff=three_body_cutoff,
            num_radial=num_radial,
            edge_feat_dim=edge_feat_dim,
            envelope_exponent=envelope_exponent,
            learnable=learnable_basis,
            distance_expansion=distance_expansion,
        )
        self.three_body_embedding = ThreebodyEmbedding(
            num_angular = num_angular, # Fourier
            max_n=max_n, max_l=max_l, cutoff=pairwise_cutoff, # Spherical Harmonics
            three_body_feat_dim = three_body_feat_dim,
            three_body_expansion = three_body_expansion,
            learnable = learnable_basis
        )
        # ====== Interaction layers ========
        interaction_block = [
            Interaction_Block(
                node_feat_dim=node_feat_dim, 
                edge_feat_dim=edge_feat_dim,
                three_body_feat_dim=three_body_feat_dim,
                num_radial=num_radial,
                num_angular=num_angular,
                dropout=dropout,
                use_bias=use_bias,
                use_smoothed_for_delta_edge=use_smoothed_for_delta_edge,
                mlp_type=graph_conv_mlp,
                norm_type=norm_type,
                activation_type=activation_type,
                enable_compile=enable_compile,
                enable_checkpoint=enable_checkpoint,
                last_block=(layer_index == num_layers - 1),
            )
            for layer_index in range(num_layers)
        ]
        self.interaction_block = nn.ModuleList(interaction_block)

        # ====== Readout layers ======== 
        self.readout_norm = get_normalization(norm_type, dim=node_feat_dim)

        self.energy_head = EnergyHead(
            feat_dim = node_feat_dim,
            hidden_dim = mlp_hidden_dims,
            output_dim = 1,
            mlp_type = "mlp",
            activation_type = activation_type,
        )
        self.magmom_head = MagmomHead(
            feat_dim = node_feat_dim,
            hidden_dim = 2 * node_feat_dim,
            output_dim = 1,
            mlp_type = "mlp",
            activation_type = activation_type,
        )
        self.force_stress_head = ForceStressHead(
            is_conservation = is_conservation,
            feat_dim = edge_feat_dim, # is_conservation == False
            hidden_dim = mlp_hidden_dims, # is_conservation == False
            output_dim = 3, # is_conservation == False
            mlp_type = "mlp", # is_conservation == False
            activation_type = activation_type, # is_conservation == False
        )
        
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"MatRIS initialized with {self.get_params()} parameters")

    def forward(
        self,
        graphs: Sequence[RadiusGraph],
        task: str = "ef",
        is_training: bool = False,
        n_real: int | Sequence[int] | Tensor | None = None,
        handle_isolated_atoms: bool = True,
    ) -> dict[str, Tensor]:
        """
        Args:
            graphs (List): a list of RadiusGraph.
            task (str): the prediction task. Can be 'e', 'em', 'ef', 'efs', 'efsm'.
            is_training: Training/fine-tuning: forces and stress keep their
                graph (create_graph) and every operator runs its plain torch
                implementation (plain_torch_scope). Pass it whenever model
                outputs are differentiated again, e.g. for force losses.
            handle_isolated_atoms: Exclude isolated atoms from learned readouts
                while retaining their atomic reference energies when the model
                has a reference-energy table.
        """
        if is_training:
            with plain_torch_scope():
                return self._forward(graphs, task, is_training, n_real, handle_isolated_atoms)
        return self._forward(graphs, task, is_training, n_real, handle_isolated_atoms)

    def _forward(self, graphs, task, is_training, n_real, handle_isolated_atoms):
        prediction = {}
        # ======== Graph processing ========
        batch_graph = process_graphs(graphs, compute_stress="s" in task)
        real_counts = self._normalize_real_counts(
            n_real,
            num_graphs=batch_graph['num_graphs'],
            device=batch_graph['atomic_numbers'].device,
        )
        readout_atom_mask = batch_graph['atom_active_mask']
        if not handle_isolated_atoms or self.reference_energy is None:
            readout_atom_mask = torch.ones_like(readout_atom_mask)
        real_atom_mask = None
        if real_counts is not None:
            real_atom_mask = torch.cat(
                [
                    torch.arange(n_atoms, device=readout_atom_mask.device)
                    < real_counts[graph_idx]
                    for graph_idx, n_atoms in enumerate(
                        batch_graph['atoms_per_graph']
                    )
                ]
            )
            readout_atom_mask = readout_atom_mask & real_atom_mask
        batch_graph['readout_atom_mask'] = readout_atom_mask

        # Ragged single-system inference skips line-graph layers altogether
        # when no real triplets exist. Dummy triplets in padded graphs must not
        # activate those layers: their biases/residuals also update real bonds
        # that have no incident triplets. Preserve that skip independently for
        # each graph in a batch, using only device-side predicates so the same
        # captured graph can disconnect and reconnect atoms during replay.
        line_graph = batch_graph['line_graph_dict']
        if len(line_graph['line_graph']) and (
            real_counts is not None or batch_graph['num_graphs'] > 1
        ):
            line_atoms = line_graph['atom_list']
            if real_counts is not None:
                real_triplets = real_atom_mask[line_atoms]
            else:
                real_triplets = torch.ones_like(line_atoms, dtype=torch.bool)
            if batch_graph['num_graphs'] == 1:
                has_real_line_graph = real_triplets.any()
            else:
                atom_segment = batch_graph['atom_segment']
                triplets_per_graph = torch.zeros(
                    batch_graph['num_graphs'],
                    dtype=torch.int32,
                    device=line_atoms.device,
                ).scatter_add_(
                    0, atom_segment[line_atoms], real_triplets.to(torch.int32)
                )
                undirected_atom = batch_graph['atom_graph_dict']['target_index'][
                    batch_graph['undirected2directed']
                ]
                has_real_line_graph = (
                    triplets_per_graph.gt(0)[atom_segment[undirected_atom]]
                    .unsqueeze(1)
                )
            batch_graph['_has_real_line_graph'] = has_real_line_graph

        # ======== Feature embedding ========
        node_feat = self.embed_atoms(batch_graph['atomic_numbers'])
        edge_feat, smooth_weight = self.edge_embedding(graphs=batch_graph) # pairwise feature init
        threebody_feat = None 
        if len(batch_graph['line_graph_dict']['line_graph']) != 0:
            threebody_feat = self.three_body_embedding(graphs=batch_graph) # three body feature init
        
        # ======== Interaction Block =======
        # InferenceConfig.saved_tensor_recompute (inference only): the blocks'
        # cheap saved activations are rebuilt in the backward instead of kept.
        recompute = "off" if is_training else _saved_recompute.mode_for(
            self, len(batch_graph['line_graph_dict']['line_graph']))
        _saved_recompute.check_model(self, recompute)
        with _saved_recompute.forward_scope(recompute):
            # Only the direct force head (readout.py) reads the last block's edge features.
            node_feat, edge_feat, threebody_feat = self.forward_features(
                batch_graph, node_feat, edge_feat, threebody_feat, smooth_weight,
                edge_output=not self.force_stress_head.is_conservation,
            )
        if recompute != "off":
            # Unused by E/F/S: drop the references before the backward (the
            # last block computed no dead update for them).
            threebody_feat = None
            if self.force_stress_head.is_conservation:
                edge_feat = None
        
        # ======== Readout Block =======
        # One readout at every rank count (``atomic_energy_rows``): the learned
        # per-atom energies plus each real atom's reference energy, summed here
        # per graph (P>1: over the owned rows, then a global SUM).
        node_feat = self.normalize_readout(node_feat)
        per_atom_reference = self._per_atom_reference()
        energies = self.atomic_energy_rows(
            node_feat, batch_graph['atomic_numbers'], readout_atom_mask, real_mask=real_atom_mask,
            reference=per_atom_reference)
        total_energy = self._graph_sum(energies, batch_graph)

        force_stress_dict = self.force_stress_head(
            batch_graph = batch_graph, 
            compute_force="f" in task,
            compute_stress="s" in task,
            total_energy = total_energy, 
            node_feat = node_feat, 
            edge_feat = edge_feat, 
            is_training = is_training)
        prediction.update(force_stress_dict)
        
        if "m" in task:
            magmom = self.magmom_head(batch_graph = batch_graph, node_feat = node_feat)
            prediction["m"] = magmom
        
        # Sync-free per-graph atom counts on device (avoid torch.tensor(list) H2D).
        atoms_per_graph_tensor = segment_count(
            batch_graph['atom_segment'], batch_graph['num_graphs']
        ).to(torch.int32)
        if self.is_intensive:
            denom = atoms_per_graph_tensor
            if real_counts is not None:
                # padded path: normalize by the real atom count, not N+dummy
                denom = real_counts
            prediction["e"] = (total_energy / denom).to(energies.dtype)
        else:
            prediction["e"] = total_energy.to(energies.dtype)

        prediction["atoms_per_graph"] = atoms_per_graph_tensor
        ref_energy = 0
        if per_atom_reference:
            # The reference part of "e" (the same gathered weights), reported alone.
            references = self.reference_energy.fc.weight[0, batch_graph['atomic_numbers'].long() - 1]
            if real_atom_mask is not None:
                references = torch.where(real_atom_mask, references, 0)
            ref_energy = self._graph_sum(references, batch_graph)
            if self.is_intensive:
                ref_energy = ref_energy / denom
            ref_energy = ref_energy.to(references.dtype)
        elif self.reference_energy is not None:
            # A table whose intensive flag differs from the model's (the MPtrj/
            # OMat/MPA initializers set it on an extensive model) keeps its own
            # composition readout added to "e".
            ref_energy = self.reference_energy(graphs, n_real=real_counts)
            prediction["e"] = prediction["e"] + ref_energy
        prediction["ref_energy"] = ref_energy
        return prediction

    def forward_features(
        self, batch_graph, node_feat, edge_feat, threebody_feat, smooth_weight, *,
        runtime=None, edge_output=True,
    ):
        """The shared interaction forward for local and partitioned execution.

        A runtime changes tensor transport and differentiation boundaries, not
        the model's layer order or attention/refinement formulas. The ordinary
        path keeps its tensors, module dispatch, checkpointing and CUDA graphs;
        it does not allocate a distributed autograd backend or communicate.
        ``edge_output=False``: nothing reads the edge features returned after the
        last block (conservative forces/stress); that block skips its atom-refinement
        edge update and returns its input bonds.
        """
        for index, mp_layer in enumerate(self.interaction_block):
            options = ({} if runtime is None else
                       {"runtime": runtime.for_block(index, mp_layer)})
            node_feat, edge_feat, threebody_feat = mp_layer(
                batch_graph=batch_graph,
                node_feat=node_feat,
                edge_feat=edge_feat,
                threebody_feat=threebody_feat,
                smooth_weight=smooth_weight,
                # Only the final block's edge output can be unread (a misplaced last_block flag
                # must not drop an earlier block's bond update).
                edge_output=edge_output or index + 1 < len(self.interaction_block),
                **options,
            )
        return node_feat, edge_feat, threebody_feat

    def embed_atoms(self, atomic_numbers: Tensor, *, module_forward=None) -> Tensor:
        """Embed actual atomic numbers (the checkpoint embedding uses H=0)."""
        return self.atom_embedding(atomic_numbers - 1, module_forward=module_forward)

    def normalize_readout(self, nodes: Tensor, *, module_forward=None) -> Tensor:
        """Shared readout normalization, retaining normal dispatch by default."""
        return (self.readout_norm(nodes) if module_forward is None
                else module_forward(self.readout_norm, nodes))

    def atomic_energy_rows(
        self, normalized: Tensor, atomic_numbers: Tensor, active: Tensor, *,
        module_forward=None, real_mask: Tensor | None = None, reference: bool = True,
    ) -> Tensor:
        """Per-atom energies of normalized readout features: the one readout of P=1 and P>1.

        Each row is the learned contribution (zero where ``active`` is False:
        isolated or padding atoms) plus the atom's elemental reference energy
        (``AtomRef`` weights, gathered by atomic number). ``real_mask`` zeroes
        whole rows that are not real atoms (padding and capacity dummies),
        reference included. Sums of these rows give the extensive energy; the
        reference terms do not depend on positions or strain, so forces and
        stress are those of the learned part. ``reference=False`` leaves the
        reference out (``forward`` with a table whose intensive flag differs).
        """
        energies = self.energy_head.atomic_energies(
            normalized, atom_mask=active, module_forward=module_forward,
        ).reshape(-1)
        if reference and self.reference_energy is not None:
            energies = energies + self.reference_energy.fc.weight[0, atomic_numbers.long() - 1]
        if real_mask is not None:
            energies = torch.where(real_mask.reshape(-1), energies, 0)
        return energies

    def atomic_energy(
        self, nodes: Tensor, atomic_numbers: Tensor, active: Tensor, *,
        module_forward=None, real_mask: Tensor | None = None,
    ) -> Tensor:
        """Extensive energy of the owned rows (P>1), before the global reduction.

        ``atomic_energy_rows`` of the normalized features, summed: the P>1
        evaluator adds only the owned mask (``real_mask``) and the SUM over
        ranks.
        """
        normalized = self.normalize_readout(nodes, module_forward=module_forward)
        return self.atomic_energy_rows(normalized, atomic_numbers, active, module_forward=module_forward,
                                       real_mask=real_mask).sum()

    def _per_atom_reference(self) -> bool:
        """Whether the reference table's per-atom weights sum to this model's reference energy."""
        return (self.reference_energy is not None
                and bool(self.reference_energy.is_intensive) == bool(self.is_intensive))

    def _graph_sum(self, rows: Tensor, batch_graph) -> Tensor:
        """Per-graph sums of per-atom rows, in FP64 (the caller rounds once).

        With the reference energies in the rows the totals are large (their
        FP32 ulp over N exceeds the ulp of the energy per atom), so they stay
        FP64 through the normalization. FP64 accumulation also makes the result
        independent of the reduction order: a static-shape graph with trailing
        dummy rows (exact zeros) gives the same bits as the unpadded graph, and
        a batch's segment sum (``index_add_``) loses nothing to its serial or
        atomic order. The gradient of a sum is 1, so forces and stress are
        unaffected.
        """
        rows64 = rows.double()
        if batch_graph['num_graphs'] == 1:
            return rows64.sum().reshape(1)
        return self.energy_head.pooling(
            rows64.unsqueeze(1), batch_graph['atom_segment'], num_segment=batch_graph['num_graphs'],
        ).view(-1)

    @staticmethod
    def _normalize_real_counts(
        n_real: int | Sequence[int] | Tensor | None,
        *,
        num_graphs: int,
        device: torch.device,
    ) -> Tensor | None:
        """Return one device-side real-atom count per graph."""
        if n_real is None:
            return None
        if isinstance(n_real, Tensor):
            counts = n_real.to(device=device, dtype=torch.int32).reshape(-1)
            if counts.numel() == 1 and num_graphs != 1:
                counts = counts.expand(num_graphs)
            if counts.numel() != num_graphs:
                raise ValueError(
                    f"n_real must contain {num_graphs} counts, got {counts.numel()}"
                )
            return counts
        if isinstance(n_real, int):
            if n_real <= 0:
                raise ValueError("n_real counts must be positive")
            return torch.full(
                (num_graphs,), n_real, dtype=torch.int32, device=device
            )
        else:
            counts = list(n_real)
            if len(counts) != num_graphs:
                raise ValueError(
                    f"n_real must contain {num_graphs} counts, got {len(counts)}"
                )
        if any(count <= 0 for count in counts):
            raise ValueError("n_real counts must be positive")
        return torch.stack(
            [
                torch.full((), count, dtype=torch.int32, device=device)
                for count in counts
            ]
        )
    
    def get_params(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_dict(cls, dct: dict):
        config = dict(dct["config"])
        config.pop("enable_compile", None)
        config.pop("enable_checkpoint", None)
        model = cls(**config)
        model.load_state_dict(dct["state_dict"])
        return model
    
    @classmethod
    def load(
        cls,
        model_path: str = None,
        model_name: str = "matris_10m_oam",
        device: str | None = None,
        enable_compile: bool = False,
        enable_checkpoint: bool | None = False,
    ):
        """Load pretrained model.

        Args:
            model_path: Local checkpoint path. If omitted, downloads the named checkpoint.
            model_name: Pretrained model name to load when model_path is omitted.
            device: Device to load the model on.
            enable_compile: Whether to compile interaction blocks.
            enable_checkpoint: False disables activation checkpointing for
                inference performance, True forces checkpointing on, and None
                uses the automatic threshold.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_path is None:
            model_name = model_name.lower()
            supported_models = ["matris_10m_oam", "matris_10m_mp"]
            if model_name not in supported_models:
                raise ValueError(
                    f"Unsupported model_name: {model_name}. Supported models are: {supported_models}"
                )

            cache_dir = os.path.expanduser("~/.cache/matris")
            os.makedirs(cache_dir, exist_ok=True)

            checkpoint_files = {
                "matris_10m_omat": "MatRIS_10M_OMAT.pth.tar",
                "matris_10m_oam": "MatRIS_10M_OAM.pth.tar",
                "matris_10m_mp": "MatRIS_10M_MP.pth.tar",
                "matris_6m_mp": "MatRIS_6M_MP.pth.tar",
            }

            DOWNLOAD_URLS = {
                "matris_10m_omat": "",  # TODO
                "matris_10m_oam": "https://figshare.com/ndownloader/files/59142728",
                "matris_10m_mp": "https://figshare.com/ndownloader/files/59143058",
                "matris_6m_mp": "",  # TODO
            }

            ckpt_filename = checkpoint_files[model_name]
            ckpt_path = os.path.join(cache_dir, ckpt_filename)

            if not os.path.exists(ckpt_path):
                url = DOWNLOAD_URLS.get(model_name)
                if not url:
                    raise ValueError(f"No download URL provided for model: {model_name}")

                print(f"Checkpoint not found, downloading to {ckpt_path} ...")
                torch.hub.download_url_to_file(url, ckpt_path)
        else:
            ckpt_path = model_path

        ckpt_state = torch.load(
            ckpt_path, 
            map_location=torch.device("cpu"), 
            weights_only=False
        )
        model = MatRIS.from_dict(ckpt_state)
        model.enable_compile = enable_compile
        model.enable_checkpoint = enable_checkpoint
        model.config["enable_compile"] = enable_compile
        model.config["enable_checkpoint"] = enable_checkpoint
        for mp_layer in model.interaction_block:
            mp_layer.enable_compile = enable_compile
            mp_layer.enable_checkpoint = enable_checkpoint
        model = model.to(device)
        print(f"Loading successfully, running on {device}.")
        
        return model
