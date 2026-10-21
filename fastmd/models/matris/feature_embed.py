from __future__ import annotations

from fastmd.config import env_value
from fastmd.engine import policy as _policy
from fastmd.engine import recorder as _recorder
from fastmd.precision import precision_cache_key


import torch
from torch import Tensor, nn
import torch.nn.functional as F

from fastmd.models.matris.basis_function import (
    FourierExpansion, 
    BesselExpansion, 
    GaussianExpansion,
    GaussianExpansion,
    SphericalExpansion,
    SinusoidalTimeExpansion,
    angular_basis,
    radial_basis,
)
from fastmd.engine.cache_provenance import CAPTURE as _CAPTURE, retain_under_capture
from fastmd.models.matris.functions import _routed, get_normalization, SwishLayer, aggregate
from fastmd.engine.kernels import triton_wide_gemm as _wide

_FUSED_THREEBODY_LINEAR_ENABLED = env_value("MATRIS_FUSED_THREEBODY_LINEAR", "1") != "0"
_FUSED_THREEBODY_LINEAR_EAGER = env_value("MATRIS_FUSED_THREEBODY_LINEAR_EAGER", "0") == "1"
_FUSED_THREEBODY_LINEAR_MIN_ROWS = int(env_value("MATRIS_FUSED_THREEBODY_LINEAR_MIN_ROWS", "60000"))
_FUSED_THREEBODY_BASIS_ENABLED = env_value("MATRIS_FUSED_THREEBODY_BASIS", "1") != "0"
_FUSED_THREEBODY_BASIS_EAGER = env_value("MATRIS_FUSED_THREEBODY_BASIS_EAGER", "0") == "1"
_FUSED_THREEBODY_BASIS_MIN_ROWS = int(env_value("MATRIS_FUSED_THREEBODY_BASIS_MIN_ROWS", "60000"))
_UNDIRECTED_EDGE_INIT = env_value("MATRIS_UNDIRECTED_EDGE_INIT", "1") != "0"


def _module_forward(module: nn.Module, value: Tensor) -> Tensor:
    """Default policy: retain the original module's normal dispatch and hooks (a frozen nn.Linear without hooks
    through the wide GEMM dispatch, see ``functions._routed``; same call depth)."""
    if type(module) is nn.Linear and _routed(module):
        return _wide._linear(value, module.weight, module.bias)
    return module(value)


# Deprecated aliases of fastmd.engine.policy.set_fusion_scope, kept for one version.
# Since step 2.2 both families share one fusion scope: setting either sets both, and
# nested set/restore pairs must be restored in reverse (LIFO) order.
def set_fused_feature_graph_optimizations(enabled: bool) -> tuple[bool, bool]:
    return (_policy.set_fusion_scope(enabled),) * 2


def restore_fused_feature_graph_optimizations(state: tuple[bool, bool]) -> None:
    _policy.set_fusion_scope(state[0])


def _use_fused_threebody_linear(basis: Tensor) -> bool:
    if not _FUSED_THREEBODY_LINEAR_ENABLED:
        return False
    if not (basis.is_cuda and basis.dim() == 2):
        return False
    if basis.shape[0] < _FUSED_THREEBODY_LINEAR_MIN_ROWS:
        return False
    return _FUSED_THREEBODY_LINEAR_EAGER or _policy.fusion_scope_active()


def _use_fused_threebody_basis(
    unit_vec: Tensor,
    target_index: Tensor,
    source_index: Tensor,
    freqs: Tensor,
) -> bool:
    if not _FUSED_THREEBODY_BASIS_ENABLED:
        return False
    if not (unit_vec.is_cuda and target_index.is_cuda and source_index.is_cuda and freqs.is_cuda):
        return False
    if unit_vec.dim() != 2 or unit_vec.shape[1] != 3:
        return False
    if target_index.numel() != source_index.numel() or target_index.numel() < _FUSED_THREEBODY_BASIS_MIN_ROWS:
        return False
    if freqs.requires_grad or freqs.numel() != 3:
        return False
    return _FUSED_THREEBODY_BASIS_EAGER or _policy.fusion_scope_active()


def _record_combined_decline(rec, reason, basis) -> None:
    rec.fallback("combined_threebody_linear", "cached", reason,
                 {"rows": basis.shape[0], "width": basis.shape[-1]})


class AtomTypeEmbedding(nn.Module):
    """Encode an atom by its atomic number using 'nn.Embedding'."""
    
    def __init__(self, atom_feat_dim: int, max_num_elements: int = 94):
        """
        Args:
            atom_feature_dim (int): dimension of atomic embedding.
            max_num_elements (int): maximum number of elements in the dataset.
                Default = 94
        """
        super().__init__()
        self.embedding = nn.Embedding(max_num_elements, atom_feat_dim)
        self.atom_swish_layer = SwishLayer(input_dim = atom_feat_dim,
                                           output_dim = atom_feat_dim,
                                           bias=False)

    def forward(self, atom_type: Tensor, *, module_forward=None) -> Tensor:
        """
        Args:
            atom_type (Tensor): [nAtoms, 1].
        """
        apply = _module_forward if module_forward is None else module_forward
        node_feat = apply(self.embedding, atom_type)
        node_feat = apply(self.atom_swish_layer, node_feat)
        return node_feat


class EdgeBasisEmbedding(nn.Module):
    """Embed edge length using 'Bessel' or 'Gaussian'."""
    
    def __init__(
        self,
        pairwise_cutoff: float = 6,
        three_body_cutoff: float = 4,
        num_radial: int = 7,
        edge_feat_dim: int = 128,
        envelope_exponent: int = 8,
        distance_expansion: str = "Bessel",
        learnable: bool = False,
    ):
        """
        Args:
            pairwise_cutoff (float): The cutoff for pairwise interaction.
                Default = 6
            three_body_cutoff (float): The cutoff for three-body interaction. 
                Default = 4
            num_radial (int): The number of radial. 
                Default = 7
            envelope_exponent (int): envelope exponent of Envelope function.
                Default = 8
            distance_expansion (str): The function of basis, "Bessel" or "Gaussian".
                Default = "Bessel"
            learnable(bool): Whether the frequency in bessel expansion is learnable.
                Default = False
        """
        super().__init__()

        self.distance_expansion = distance_expansion
        if self.distance_expansion.lower() == "bessel":
            self.pairwise_rbf_expansion = BesselExpansion(
                num_radial=num_radial,
                cutoff=pairwise_cutoff,
                envelope_exponent=envelope_exponent,
                learnable=learnable,
            )
            self.threebody_rbf_expansion = BesselExpansion(
                num_radial=num_radial,
                cutoff=three_body_cutoff,
                envelope_exponent=envelope_exponent,
                learnable=learnable,
            )
        elif self.distance_expansion.lower() == "gaussian":
            self.pairwise_rbf_expansion = GaussianExpansion(
                start=0.0,
                stop=pairwise_cutoff,
                num_gaussians=num_radial,
                basis_width_scalar=2.0,
            )
            self.threebody_rbf_expansion = GaussianExpansion(
                start=0.0,
                stop=three_body_cutoff,
                num_gaussians=num_radial,
                basis_width_scalar=2.0,
            )
        else:
            raise NotImplementedError

        self.edge_linear1 = nn.Linear(
            in_features=num_radial, out_features=edge_feat_dim, bias=False
        )
        self.edge_linear2 = nn.Linear(
            in_features=edge_feat_dim, out_features=edge_feat_dim, bias=False
        )
        self.swish_layer = SwishLayer(input_dim = edge_feat_dim,
                                           output_dim = edge_feat_dim,
                                           bias=False)
        self.edge_init_norm = get_normalization(name="layer", dim=edge_feat_dim)

        
    def forward(
        self,
        graphs, 
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Args:
            center_pos (Tensor): coord of center atoms.
            neighbor_pos (Tensor): coordinates of neighbor atoms.
            undirected2directed (Tensor): mapping from undirected vectors to one of its
                directed vectors. Following repo: https://github.com/CederGroupHub/chgnet
            image (Tensor): the periodic image specifying the location of neighbor atoms.
            lattice (Tensor): the lattice matrix of structure.
        """
         
        # From directed edge to undirected edge, this motivation is inspired by https://github.com/CederGroupHub/chgnet
        unique_edge_lengths = torch.index_select( graphs['edge_lengths'], 0, graphs['undirected2directed'] ) #[nEdges/2]

        edge_feat, pairwise_rbf, threebody_rbf = self.forward_lengths(
            unique_edge_lengths,
            directed2undirected=(None if _UNDIRECTED_EDGE_INIT else graphs['directed2undirected']),
        )
        smooth_weight = {"atom graph": pairwise_rbf, "line graph": threebody_rbf}
        return edge_feat, smooth_weight

    def forward_lengths(
        self, lengths: Tensor, *, module_forward=None,
        directed2undirected: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed one row per undirected bond, independent of graph ownership.

        A graph adapter may supply its directed map to preserve the optional
        legacy directed-initialization path. Owner-complete distributed bond
        rows need no directed expansion for this pointwise initialization.
        """
        apply = _module_forward if module_forward is None else module_forward
        pairwise_rbf = radial_basis(self.pairwise_rbf_expansion, lengths)
        threebody_rbf = radial_basis(self.threebody_rbf_expansion, lengths)
        edge_feat_undirect = apply(self.edge_linear1, pairwise_rbf)
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("edge_init", "undirected"
                       if _UNDIRECTED_EDGE_INIT or directed2undirected is None
                       else "directed_aggregate", {"rows": lengths.shape[0]})
        if _UNDIRECTED_EDGE_INIT or directed2undirected is None:
            edge_feat = apply(self.edge_linear2, edge_feat_undirect)
            edge_feat = apply(self.edge_init_norm, edge_feat)
        else:
            # [edges, dim] -> [2*edges, dim]
            edge_feat_direct = torch.index_select(edge_feat_undirect, 0, directed2undirected)
            edge_feat_direct = apply(self.edge_linear2, edge_feat_direct)
            edge_feat_direct = apply(self.edge_init_norm, edge_feat_direct)
            # Aggregate to bond_feas_ude
            edge_feat = aggregate(data=edge_feat_direct,
                                                     segment=directed2undirected,
                                                     bin_count=None,
                                                     average=True,
                                                     num_segment=lengths.shape[0])
        edge_feat = apply(self.swish_layer, edge_feat)
        return edge_feat, pairwise_rbf, threebody_rbf


class ThreebodyEmbedding(nn.Module):
    """Embed threebody using 'Fourier' or 'Sphere Harmonic'."""

    def __init__(self, 
                 num_angular: int = 7, # Fourier
                 max_n: int=4, max_l: int=4, cutoff: int=6,  #SH
                 three_body_feat_dim: int = 128,
                 three_body_expansion: str = "fourier",
                 learnable: bool = True):
        super().__init__()
        
        self.three_body_expansion = three_body_expansion.lower()
        if self.three_body_expansion == "fourier":
            max_f = (num_angular - 1) // 2
            self.expansion = FourierExpansion(
                max_f=max_f, learnable=learnable
            )
            self.angle_embedding = nn.Linear(
                in_features=num_angular, out_features=three_body_feat_dim, bias=False
            )
        elif self.three_body_expansion == "sh":
            self.expansion = SphericalExpansion(max_n=max_n, max_l=max_l, cutoff=cutoff)
            self.angle_embedding = nn.Linear(
                in_features=max_n * max_l, out_features=three_body_feat_dim, bias=False
            ) 
        else:
            raise NotImplementedError

        self.swish_layer = SwishLayer(input_dim = three_body_feat_dim,
                                    output_dim = three_body_feat_dim,
                                    bias=False)
        self._combined_swish_cache_key = None
        self._combined_swish_weight = None

    def _combined_swish_linear_weight(
        self, basis: Tensor, *, apply_profile: bool = True, dispatcher=None,
    ) -> Tensor | None:
        """Build the shared frozen projection, optionally under an explicit policy.

        Single-device callers retain their existing global/profile qualification.
        An executor may bypass only that qualification after admitting its own
        tensor/module contract; frozen weights and cache safety remain here.
        This does not enable the indexed Fourier basis kernel. With a
        ``dispatcher`` (unified engine) its threebody_linear policy replaces the
        module switches of the profile qualification.
        """
        if type(apply_profile) is not bool:
            raise TypeError("apply_profile must be bool")
        if (self.three_body_expansion != "fourier"
                or apply_profile and not (_use_fused_threebody_linear(basis) if dispatcher is None
                                          else dispatcher.use_threebody_linear(basis))):
            rec = _recorder._ACTIVE
            if rec is not None:
                _record_combined_decline(
                    rec, "not_fourier" if self.three_body_expansion != "fourier"
                    else "profile_gate", basis)
            return None
        if not isinstance(self.angle_embedding, nn.Linear):
            rec = _recorder._ACTIVE
            if rec is not None:
                _record_combined_decline(rec, "angle_embedding_not_linear", basis)
            return None
        swish_linear = self.swish_layer.linear
        if self.angle_embedding.bias is not None or swish_linear.bias is not None:
            rec = _recorder._ACTIVE
            if rec is not None:
                _record_combined_decline(rec, "projection_has_bias", basis)
            return None
        angle_w = self.angle_embedding.weight
        swish_w = swish_linear.weight
        if angle_w.requires_grad or swish_w.requires_grad:
            rec = _recorder._ACTIVE
            if rec is not None:
                _record_combined_decline(rec, "trainable_projection", basis)
            return None
        if angle_w.device != basis.device or swish_w.device != basis.device:
            rec = _recorder._ACTIVE
            if rec is not None:
                _record_combined_decline(rec, "device_mismatch", basis)
            return None
        if angle_w.dtype != basis.dtype or swish_w.dtype != basis.dtype:
            rec = _recorder._ACTIVE
            if rec is not None:
                _record_combined_decline(rec, "dtype_mismatch", basis)
            return None
        autocast_enabled = torch.is_autocast_enabled(basis.device.type)
        key = (
            id(angle_w),
            id(swish_w),
            angle_w.data_ptr(),
            swish_w.data_ptr(),
            angle_w._version,
            swish_w._version,
            tuple(angle_w.shape),
            tuple(swish_w.shape),
            tuple(angle_w.stride()),
            tuple(swish_w.stride()),
            basis.device,
            basis.dtype,
            # The same frozen tensors can have been warmed by another
            # executor under TF32/high. Such a cache is not a highest-precision
            # projection merely because its output tensor has dtype float32.
            # Reported and effective cuBLAS math; never raises on mixed APIs.
            precision_cache_key(),
            autocast_enabled,
            torch.get_autocast_dtype(basis.device.type) if autocast_enabled else None,
        )
        if self._combined_swish_cache_key != key:
            if basis.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                rec = _recorder._ACTIVE
                if rec is not None:
                    _record_combined_decline(rec, "cold_cache_under_capture", basis)
                return None
            with torch.no_grad():
                self._combined_swish_weight = torch.matmul(
                    swish_w.detach(), angle_w.detach()
                ).contiguous()
                self._combined_swish_cache_key = key
                from fastmd.engine.cache_provenance import publish_projection_cache
                publish_projection_cache(self, "combined_swish", (angle_w, swish_w))
        if not torch.compiler.is_compiling() and _CAPTURE.holder is not None:  # a capture keeps what it reads
            retain_under_capture(self._combined_swish_weight, (angle_w, swish_w))
        return self._combined_swish_weight
     
    def forward(self, graphs):
        return self.forward_indexed(
            graphs['unit_edge_vectors'],
            graphs['line_graph_dict']['target_DE_index'],
            graphs['line_graph_dict']['source_DE_index'],
            dispatcher=graphs.get("_ops"),
        )

    def forward_indexed(
        self, unit_edge_vectors: Tensor, target_de_index: Tensor,
        source_de_index: Tensor, *, module_forward=None, allow_fused: bool = True,
        combined_linear_policy=None, dispatcher=None,
    ) -> Tensor:
        """Embed indexed triplets; indices can address global or compact rows.

        The default preserves single-card indexed Fourier fusion. An explicit
        reference policy disables feature fusions without changing global flags.
        A separate executor policy can qualify the frozen linear projection
        while keeping the basis kernel disabled. Omitting it preserves the
        original meaning of allow_fused, including allow_fused=False. A
        ``dispatcher`` (unified engine) supplies the threebody_basis and
        threebody_linear policy instead of the module switches.
        """
        if (
            allow_fused and self.three_body_expansion == "fourier"
            and (_use_fused_threebody_basis if dispatcher is None
                 else dispatcher.use_threebody_basis)(
                unit_edge_vectors,
                target_de_index,
                source_de_index,
                self.expansion.frequencies,
            )
        ):
            from fastmd.models.matris.op.triton_threebody_fourier import threebody_fourier_basis

            three_body_basis = threebody_fourier_basis(
                unit_edge_vectors,
                target_de_index,
                source_de_index,
                self.expansion.frequencies,
                self.expansion.scale,
            )
            return self.forward_basis(three_body_basis, module_forward=module_forward,
                                      allow_fused=allow_fused,
                                      combined_linear_policy=combined_linear_policy,
                                      dispatcher=dispatcher)
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("threebody_basis", "torch", {
                "rows": target_de_index.numel(), "expansion": self.three_body_expansion,
                "allow_fused": allow_fused,
            })
        return self.forward_vectors(
            torch.index_select(unit_edge_vectors, 0, target_de_index),
            torch.index_select(unit_edge_vectors, 0, source_de_index),
            module_forward=module_forward, allow_fused=allow_fused,
            combined_linear_policy=combined_linear_policy, dispatcher=dispatcher,
        )

    def forward_vectors(
        self, target_unit: Tensor, source_unit: Tensor, *, module_forward=None,
        allow_fused: bool = True, combined_linear_policy=None, dispatcher=None,
    ) -> Tensor:
        """Embed already-paired unit vectors using the same basis and projection."""
        theta = torch.sum(target_unit * source_unit, dim=1) * (1 - 1e-6)
        angles = torch.acos(theta)
        # Fourier never consumes radial lengths: retain the original no-norm
        # path instead of adding a GPU reduction solely for a uniform signature.
        source_lengths = torch.norm(source_unit, dim=1) if self.three_body_expansion == "sh" else None
        basis = angular_basis(self.expansion, angles, source_lengths)
        return self.forward_basis(basis, module_forward=module_forward,
                                  allow_fused=allow_fused,
                                  combined_linear_policy=combined_linear_policy,
                                  dispatcher=dispatcher)

    def forward_basis(
        self, three_body_basis: Tensor, *, module_forward=None,
        allow_fused: bool = True, combined_linear_policy=None, dispatcher=None,
    ) -> Tensor:
        """Project a basis with the original optional frozen-weight fusion."""
        apply = _module_forward if module_forward is None else module_forward
        if combined_linear_policy is not None:
            # The explicit policy is authoritative: a declined projection
            # must not fall through to a process-global optimization switch.
            combined_weight = combined_linear_policy(self, three_body_basis)
        else:
            combined_weight = (self._combined_swish_linear_weight(three_body_basis, dispatcher=dispatcher)
                               if allow_fused else None)
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("threebody_linear", "torch_linear" if combined_weight is None
                       else "combined_cached", {
                           "rows": three_body_basis.shape[0], "width": three_body_basis.shape[-1],
                           "policy": combined_linear_policy is not None, "allow_fused": allow_fused,
                       })
        if combined_weight is None:
            threebody_feat = apply(self.angle_embedding, three_body_basis)
            threebody_feat = apply(self.swish_layer, threebody_feat)
        else:
            threebody_feat = F.linear(three_body_basis, combined_weight)
            threebody_feat = apply(self.swish_layer.act, threebody_feat)
        
        return threebody_feat
    

class ThreebodyFourierExpansion(nn.Module):
    """Encode an three-body terms using Fourier Expansion."""
    
    def __init__(self, num_angular: int = 7, learnable: bool = True):
        super().__init__()
        
        assert num_angular % 2 == 1, f"{num_angular=} must be an odd integer"
        max_f = (num_angular - 1) // 2
        self.fourier_expansion = FourierExpansion(
            max_f=max_f, learnable=learnable
        )

    def forward(self, edge_vec_ij: Tensor, edge_vec_jk: Tensor) -> Tensor:
        """
        Args:
            edge_vec_i (Tensor): normalized edge vector ij [n_angle, 3]
            edge_vec_j (Tensor): normalized edge vector jk [n_angle, 3]
        """
        theta_ijk = torch.sum(edge_vec_ij * edge_vec_jk, dim=1) * (1 - 1e-6)
        angle = torch.acos(theta_ijk) 
        return self.fourier_expansion(angle)


class three_bodySHExpansion(nn.Module):
    """Encode an three-body terms using Sphere Harmonic Expansion."""

    def __init__(self, max_n: int=4, max_l: int=4, cutoff: int=6) -> None:
        super().__init__()
        self.sbf = SphericalExpansion(max_n=max_n, max_l=max_l, cutoff=cutoff)
    
    def forward(self, edge_vec_ij: Tensor, edge_vec_jk: Tensor) -> Tensor:
        """
        Args:
            edge_vec_i (Tensor): normalized edge vector ij [n_angle, 3]
            edge_vec_j (Tensor): normalized edge vector jk [n_angle, 3]
        """
        theta_ijk = torch.sum(edge_vec_ij * edge_vec_jk, dim=1) * (1 - 1e-6)
        angle = torch.acos(theta_ijk)
        
        edge_vec_jk_len = torch.norm(edge_vec_jk, dim=1)
        angle_basis = self.sbf(edge_vec_jk_len, angle)
        return angle_basis


class TimeEmbedding(nn.Module):
    """Encode time using nonlinear"""

    def __init__(self, time_feat_dim):
        super().__init__()
        self.time_encode = SinusoidalTimeExpansion(dim = time_feat_dim)
        self.linear1 = nn.Linear(
            in_features = time_feat_dim, out_features = 2 * time_feat_dim
        )
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(
            in_features = time_feat_dim * 2, out_features = time_feat_dim
        )
     
    def forward(self, time):
        time_enc= self.time_encode(time)
        time_enc = self.act(self.linear1(time_enc))
        time_emb = self.linear2(time_enc)
        return time_emb
