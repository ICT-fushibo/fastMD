from __future__ import annotations
from typing import Literal

from fastmd.config import env_value
from fastmd.precision import triton_dot_input_precision

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn.modules import module as _torch_module
import math
from fastmd.engine import recorder as _recorder
from fastmd.engine import saved_recompute as _saved_recompute
from fastmd.engine.cache_provenance import CAPTURE as _CAPTURE, retain_under_capture
from fastmd.models.matris import projection_caches as _projection_caches  # noqa: F401  (registers MatRIS's cache groups)
from fastmd.models.matris.op import fused_silu, fused_sigmoid
from fastmd.engine.kernels import triton_wide_gemm as _wide
# The segment helpers live in the engine (the operator dispatch uses them too); the model imports them from here.
from fastmd.engine.kernels.segment import IndexedProjectionConfig, Dimwise_softmax, aggregate, segment_count

# Row-count threshold above which the fused Triton LayerNorm+activation (Fusion C)
# is used instead of eager nn.LayerNorm+act.
_LN_FUSE_THRESH = int(env_value("MATRIS_GATED_LN_FUSE_MIN_ROWS", "10000"))
_FUSED_GATED_LN_RESIDUAL = env_value("MATRIS_FUSED_GATED_LN_RESIDUAL", "1") != "0"
# CUDA FusedSiLU/FusedSigmoid custom ops; off only in fastmd.models.matris.model.plain_torch_scope.
_FUSED_ACTIVATIONS = True
_INDEXED_CAT_LINEAR_FUSE = env_value("MATRIS_FUSED_INDEXED_CAT_LINEAR", "0") == "1"
_INDEXED_CAT_LINEAR_BACKEND = env_value(
    "MATRIS_FUSED_INDEXED_CAT_LINEAR_BACKEND", "decomposed_indexed_silu"
).lower()
_INDEXED_CAT_LINEAR_MIN_ROWS = int(
    env_value("MATRIS_INDEXED_CAT_LINEAR_MIN_ROWS", "20000")
)
# Step 2.5b (D8): indexed projections whose parts differ in width (e.g. [32|128|32|32]).
_MIXED_WIDTH_CAT = env_value("MATRIS_MIXED_WIDTH_CAT", "1") == "1"
# Step 2.5b (D13a): the fused gated LayerNorm adds the second linears' bias (no GEMM bias epilogue).
_GATED_LN_BIAS_FOLD = env_value("MATRIS_GATED_LN_BIAS_FOLD", "1") == "1"
# C2 (step 2.5b D13b): the narrow tail (both second linears + the gated-LN finish) in one kernel per direction.
_FUSED_GATED_TAIL = env_value("MATRIS_FUSED_GATED_TAIL", "1") == "1"
_INDEXED_CAT_LINEAR_BACKENDS = {
    "decomposed_indexed_silu",
}
if _INDEXED_CAT_LINEAR_FUSE and _INDEXED_CAT_LINEAR_BACKEND not in _INDEXED_CAT_LINEAR_BACKENDS:
    raise ValueError(
        "unsupported MATRIS_FUSED_INDEXED_CAT_LINEAR_BACKEND="
        f"{_INDEXED_CAT_LINEAR_BACKEND!r}; expected one of "
        f"{sorted(_INDEXED_CAT_LINEAR_BACKENDS)}"
    )


def set_fused_activations(enabled: bool) -> bool:
    global _FUSED_ACTIVATIONS
    previous, _FUSED_ACTIVATIONS = _FUSED_ACTIVATIONS, bool(enabled)
    return previous


class FusedSiLU(torch.nn.Module):
    """Fused Sigmoid Linear Unit."""

    def __init__(self) -> None:
        """Initialize a fused SiLU."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if x.device.type == "cuda" and _FUSED_ACTIVATIONS:
            return fused_silu(x)
        else:
            rec = _recorder._ACTIVE
            if rec is not None:
                rec.record("silu", "torch", {"shape": list(x.shape)})
            return torch.nn.functional.silu(x) 

class FusedSigmoid(torch.nn.Module):
    """Fused Sigmoid Linear Unit."""

    def __init__(self) -> None:
        """Initialize a fused SiLU."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if x.device.type == "cuda" and _FUSED_ACTIVATIONS:
            return fused_sigmoid(x)
        else:
            rec = _recorder._ACTIVE
            if rec is not None:
                rec.record("sigmoid", "torch", {"shape": list(x.shape)})
            return torch.nn.functional.sigmoid(x)

def get_activation(name: str) -> nn.Module:
    """Return an activation function"""
    activation_map = {
        "relu": nn.ReLU,
        "silu": FusedSiLU,  # Using fused version for better performance
        "gelu": nn.GELU,
        "softplus": nn.Softplus,
        "sigmoid": FusedSigmoid,  # Using fused version for better performance
        "tanh": nn.Tanh,
    }
    
    name_lower = name.lower()
    if name_lower not in activation_map:
        raise NotImplementedError(
            f"Activation '{name}' is not implemented. "
            f"Supported activations: {list(activation_map.keys())}"
        )
    return activation_map[name_lower]()

def get_normalization(name: str, dim: int | None = None) -> nn.Module | None:
    """Return an normalization function"""
    if name is None:
        return None
        
    normalization_map = {
        "layer": nn.LayerNorm(dim),
        "rms": nn.RMSNorm(dim), # torch >= 2.6.0
        "batch": nn.BatchNorm1d(dim),
    }
    name_lower = name.lower()
    return normalization_map[name_lower]

class SwishLayer(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 128,
        bias: bool = True,
    ) -> None:
        """
        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            bias: Whether to use bias in the linear layer. Default: True.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.act = get_activation("silu")
    
    def forward(self, feas: Tensor) -> Tensor:
        """
        Args:
            feas: shape (feas_num, in_dim)
            
        Returns:
            output: shape (feas_num, out_dim)
        """
        return self.act(_apply_layer(self.linear, feas))


class MLP(nn.Module):
        
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int | Sequence[int] | None = (128, 128),
        output_dim: int = 128,
        dropout: float = 0.0,
        activation: Literal["silu", "relu", "tanh", "gelu"] = "silu",
        bias: bool = True,
        use_fp16: bool = False,
    ):
        """Initialize the MLP layer.
        Args:
            input_dim: Dimension of input features.
            hidden_dim: Number of hidden units. Can be an integer for a single
                hidden layer, a sequence of integers for multiple hidden layers,
                or None for no hidden layers. Default: (128, 128).
            output_dim: Dimension of output predictions. Default: 128.
            dropout: Dropout rate applied before each linear layer. Default: 0.0.
            activation: Activation function. Supported: "relu", "silu", "tanh", "gelu".
            bias: Whether to use bias in linear layers. Default: True.
            use_fp16: Whether to use mixed precision (FP16). Default: False.
        """
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"Dropout rate must be in [0.0, 1.0), got {dropout}")
        
        self.use_fp16 = use_fp16
        activation_func = get_activation(activation)

        layers = []
        if hidden_dim in (None, 0):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(input_dim, output_dim, bias=bias))
        elif isinstance(hidden_dim, int):
            # Single hidden layer
            layers.extend([
                nn.Linear(input_dim, hidden_dim, bias=bias),
                activation_func,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim, bias=bias),
            ])
        elif isinstance(hidden_dim, Sequence):
            # Multiple hidden layers
            layers.extend([
                nn.Linear(input_dim, hidden_dim[0], bias=bias),
                activation_func,
            ])
            # Additional hidden layers
            for i in range(len(hidden_dim) - 1):
                layers.extend([
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim[i], hidden_dim[i + 1], bias=bias),
                    activation_func,
                ])
            # Output layer
            layers.extend([nn.Dropout(dropout), nn.Linear(hidden_dim[-1], output_dim, bias=bias)])
        else:
            raise TypeError(
                f"hidden_dim must be an integer, sequence of integers, or None, "
                f"got {type(hidden_dim).__name__}"
            )
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, feas: Tensor, bias: bool = True) -> Tensor:
        """
            Args:
                feas: Input tensor of shape (features, input_dim)
                bias: False leaves out the last linear's bias (D13a: a GatedMLP's fused
                    gated LayerNorm adds it; the GatedMLP folds FP32 layers only)
            Returns:
                Output tensor of shape (features, output_dim)
        """
        if self.use_fp16 and feas.is_cuda:
            rec = _recorder._ACTIVE
            if rec is not None:
                rec.record("mlp", "autocast_fp16", {"rows": feas.shape[0], "width": feas.shape[-1]})
            with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
                out = self.layers(feas) if bias else _run_layers(list(self.layers), feas, True)
            out = out.to(torch.float32)
        else:
            scope = _saved_recompute.active()
            if scope is not None and len(self.layers) > 1 and isinstance(self.layers[0], nn.Linear):
                source = scope.ref(feas)
                if source is not feas:
                    # Saved-tensor recompute: the input is rebuilt in the backward, so is the first
                    # linear's output (the activation's saved input).
                    precision = _layer_precision(self.layers[0], feas)  # the rebuild takes the forward's GEMM
                    first = _apply_layer(self.layers[0], feas, precision)
                    scope.register((first,), _apply_layer, self.layers[0], source, precision)
                    return _run_layers(list(self.layers)[1:], first, not bias)
            if not bias:
                return _run_layers(list(self.layers), feas, True)
            # The module call unless the wide GEMM dispatch is on and the container has no hooks
            # (torch.compile'd layers keep it: their compiled program is unchanged).
            out = _run_layers(list(self.layers), feas, False) if _unhooked(self.layers) else self.layers(feas)
        return out


def _unhooked(module: nn.Module) -> bool:
    """Whether the wide GEMM dispatch may replace calls of ``module`` (switch on, not compiling): no module hooks,
    no instance ``forward`` and no global module hooks (the op-selection recorder's own only observe)."""
    if (torch.compiler.is_compiling() or not _wide._enabled() or module._forward_hooks or module._forward_pre_hooks
            or module._backward_hooks or module._backward_pre_hooks or "forward" in module.__dict__):
        return False
    tm = _torch_module
    if not (tm._global_forward_hooks or tm._global_forward_pre_hooks or tm._global_backward_hooks
            or tm._global_backward_pre_hooks):
        return True
    exempt = _recorder.recorder_hook_ids()
    return not any(set(table) - exempt for table in (tm._global_forward_hooks, tm._global_forward_pre_hooks,
                                                      tm._global_backward_hooks, tm._global_backward_pre_hooks))


def _routed(layer: nn.Module) -> bool:
    """Whether ``_apply_layer`` calls the wide GEMM dispatch instead of the module: an exact ``nn.Linear`` without
    hooks while the switch is on (off: the module call, as before)."""
    return type(layer) is nn.Linear and _unhooked(layer)


def _apply_layer(layer: nn.Module, x: Tensor, precision=_wide.DECIDE) -> Tensor:
    """One layer of an MLP: a frozen ``nn.Linear`` (without hooks) through the wide GEMM dispatch
    (``triton_wide_gemm.linear``: 3xTF32 for the wide projections, else ``F.linear``), else the module.
    ``precision``: a recompute recipe's binding of the forward's GEMM (``_layer_precision``)."""
    if type(layer) is nn.Linear and _unhooked(layer):
        return _wide._linear(x, layer.weight, layer.bias, precision)
    return layer(x)


def _layer_precision(layer: nn.Module, x: Tensor):
    """The GEMM ``_apply_layer(layer, x)`` takes now, for a recompute recipe to bind (None: ``F.linear``, the module
    call's own when not routed)."""
    return _wide.linear_precision(x, layer.weight, layer.bias) if _routed(layer) else None


def _run_layers(layers, x: Tensor, fold: bool) -> Tensor:
    """``layers`` in order; ``fold``: the last one (a Linear) without its bias (D13a)."""
    for layer in layers[:-1] if fold else layers:
        # ``_apply_layer`` inline (host cost per layer).
        x = (_wide._linear(x, layer.weight, layer.bias) if type(layer) is nn.Linear and _unhooked(layer)
             else layer(x))
    return _wide.linear(x, layers[-1].weight) if fold else x


def _record_indexed_decline(rec, op, backend, reason, aligned) -> None:
    rec.fallback(op, backend if backend is not None else _INDEXED_CAT_LINEAR_BACKEND, reason,
                 {"rows": aligned.shape[0], "width": aligned.shape[-1]})


def _residual_decline_reason(module, feas, residual, res_weight, min_rows) -> str:
    """First failing guard of GatedMLP.forward_with_residual (recording only)."""
    if not module._fuse_ln:
        return "norm_or_activation_not_fusable"
    if not (feas.is_cuda and residual.is_cuda and res_weight.is_cuda):
        return "not_cuda"
    if residual.dim() != 2:
        return "residual_not_2d"
    if residual.shape[0] < min_rows:
        return "below_min_rows"
    if residual.shape[1] != res_weight.numel():
        return "width_mismatch"
    return "trainable_residual_weight"


def _part_widths(dim, parts: int) -> tuple:
    """Per-part input widths: ``dim`` is one width for ``parts`` equal parts,
    or the widths themselves (a tuple of ``parts`` entries)."""
    return tuple(dim) if isinstance(dim, (tuple, list)) else (dim,) * parts


def _mixed_width_allowed(widths: tuple, projection_config) -> bool:
    for width in widths[1:]:
        if width != widths[0]:
            # Not inside torch.compile'd layers (enable_compile): there every GatedMLP's split
            # frame recompiles (per module) and the RDX compile config measured +3.6%.
            enabled = _MIXED_WIDTH_CAT if projection_config is None else projection_config.mixed_width
            return enabled and not torch.compiler.is_compiling()
    return True


def _first_linear_decline_reason(module, rows, dim, parts, device, dtype,
                                 projection_config=None) -> str:
    """First failing guard of GatedMLP's split first-linear params (recording only)."""
    enabled = (_INDEXED_CAT_LINEAR_FUSE if projection_config is None
               else projection_config.enabled)
    min_rows = (_INDEXED_CAT_LINEAR_MIN_ROWS if projection_config is None
                else projection_config.min_rows)
    if not enabled:
        return "disabled"
    if module.use_fp16:
        return "fp16"
    if module.gate_norm is None or not module._fuse_ln:
        return "norm_or_activation_not_fusable"
    if rows < min_rows:
        return "below_min_rows"
    modules = module._first_linear_modules()
    if modules is None:
        return "first_layer_not_linear"
    core_linear, gate_linear = modules
    widths = _part_widths(dim, parts)
    if (len(widths) != parts or core_linear.in_features != sum(widths)
            or gate_linear.in_features != sum(widths)
            or core_linear.out_features != gate_linear.out_features):
        return "width_mismatch"
    if not _mixed_width_allowed(widths, projection_config):
        return "mixed_width_under_compile" if torch.compiler.is_compiling() else "mixed_width_disabled"
    if (core_linear.bias is None) != (gate_linear.bias is None):
        return "bias_mismatch"
    linear = [core_linear.weight, gate_linear.weight]
    if core_linear.bias is not None:
        linear += [core_linear.bias, gate_linear.bias]
    norms = [module.core_norm.weight, module.core_norm.bias,
             module.gate_norm.weight, module.gate_norm.bias]
    if any(t.requires_grad for t in linear + norms):
        return "trainable_params"
    if any(t.device != device or t.dtype != dtype for t in linear):
        return "device_or_dtype_mismatch"
    # Every guard passed: the params cache was cold during CUDA graph capture
    # (recorded separately as indexed_projection_weights/cold_cache_under_capture).
    return "cold_cache_under_capture"


def _record_residual_decline(rec, reason, residual) -> None:
    # The declined guard may be the residual's rank itself: record its shape.
    rec.fallback("gated_ln_residual", "triton", reason, {"shape": list(residual.shape)})


def _first_linear_by_parts(aligned: Tensor, nodes: Tensor, target: Tensor, source: Tensor,
                           weight: Tensor, bias: Tensor | None) -> Tensor:
    """``linear(cat([aligned, nodes[target], nodes[source]], 1), weight, bias)``
    by parts (saved-tensor recompute, deep): a narrow GEMM on the aligned rows
    plus the two node projections gathered and added, without the wide
    concatenation. Rebuilds a saved first-layer output within rounding, not
    bitwise (the forward is unchanged)."""
    dim, hidden = aligned.shape[1], nodes.shape[1]
    out = _wide.linear(aligned, weight[:, :dim], bias)
    out += _wide.linear(nodes, weight[:, dim:dim + hidden]).index_select(0, target)
    out += _wide.linear(nodes, weight[:, dim + hidden:]).index_select(0, source)
    return out


class GatedMLP(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int | Sequence[int] | None = (128, 128),
        output_dim: int = 128,
        dropout: float = 0.0,
        activation: str = "silu",
        norm_type: str = "layer",
        bias: bool = True,
        use_fp16: bool = False,
    ) -> None:
        """
        Args:
            input_dim: The input dimension.
            hidden_dim: A list of integers or a single integer representing the number 
                of hidden units in each layer of the MLP. Default: None.
            output_dim: The output dimension.
            dropout: The dropout rate. Default: 0.0.
            activation: The name of the activation function. Must be one of "relu", 
                "silu", "tanh", or "gelu". Default: "silu".
            norm_type: The name of the normalization layer to use. Must be one of 
                "layer", "rms", "batch", "group", or None. Default: "layer".
            bias: Whether to use bias in linear layers. Default: True.
            use_fp16: Whether to use mixed precision (FP16). Default: False.
        """
        super().__init__()
        self.use_fp16 = use_fp16
        self.activation_func = get_activation(activation)
        self.activation_gate = get_activation("sigmoid")
        self.gate_norm = get_normalization(name=norm_type, dim=output_dim)
        self.core_norm = get_normalization(name=norm_type, dim=output_dim)
        self.mlp_core = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
            use_fp16=use_fp16,
        )
        self.mlp_gate = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
            use_fp16=use_fp16,
        )
        # Fusion C: fuse LayerNorm+activation when the norm is LayerNorm and the
        # activations are SiLU(core)/Sigmoid(gate). Only applied to large row counts
        # (line-graph tensors), where the fused Triton kernel is a net win (~2.6x);
        # torch's LayerNorm already wins at small row counts, so guard by a threshold.
        from fastmd.models.matris.op.triton_layernorm_act import ACT_SILU
        from fastmd.models.matris.op.triton_layernorm_act import ACT_SIGMOID
        self._fuse_ln = (
            isinstance(self.core_norm, nn.LayerNorm)
            and isinstance(self.gate_norm, nn.LayerNorm)
            and isinstance(self.activation_func, FusedSiLU)
            and isinstance(self.activation_gate, FusedSigmoid)
        )
        self._core_act, self._gate_act = ACT_SILU, ACT_SIGMOID
        self._indexed_cat_linear_split_cache_key = None
        self._indexed_cat_linear_split_weights = None
        self._indexed_cat_linear_split_bias = None

    def _finish_gate(self, core_x: Tensor, gate_x: Tensor, biases=None) -> Tensor:
        """``biases``: the second linears' (core, gate) biases left out of ``core_x``/``gate_x``
        (D13a, from ``_folded_biases``, which qualified the fused gated LayerNorm; the other
        branches add them back should a guard differ)."""
        if self.gate_norm is None:
            if biases is not None:
                core_x, gate_x = core_x + biases[0], gate_x + biases[1]
            rec = _recorder._ACTIVE
            if rec is not None:
                rec.record("gated_ln", "torch", {
                    "rows": core_x.shape[0], "width": core_x.shape[-1], "norm": None,
                })
            core = self.activation_func(core_x)
            gate = self.activation_gate(gate_x)
        elif self._fuse_ln and core_x.is_cuda and core_x.shape[0] >= _LN_FUSE_THRESH:
            cn, gn = self.core_norm, self.gate_norm
            if not (
                cn.weight.requires_grad
                or cn.bias.requires_grad
                or gn.weight.requires_grad
                or gn.bias.requires_grad
            ):
                from fastmd.models.matris.op.triton_gated_ln import fused_gated_ln
                return fused_gated_ln(
                    core_x,
                    gate_x,
                    cn.weight,
                    cn.bias,
                    gn.weight,
                    gn.bias,
                    cn.eps,
                    gn.eps,
                    *(() if biases is None else biases),
                )
            from fastmd.models.matris.op.triton_layernorm_act import fused_ln_act
            if biases is not None:
                core_x, gate_x = core_x + biases[0], gate_x + biases[1]
            core = fused_ln_act(core_x, cn.weight, cn.bias, cn.eps, self._core_act)
            gate = fused_ln_act(gate_x, gn.weight, gn.bias, gn.eps, self._gate_act)
        else:
            if biases is not None:
                core_x, gate_x = core_x + biases[0], gate_x + biases[1]
            rec = _recorder._ACTIVE
            if rec is not None:
                rec.record("gated_ln", "torch", {
                    "rows": core_x.shape[0], "width": core_x.shape[-1],
                    "norm": type(self.core_norm).__name__,
                })
            core = self.activation_func(self.core_norm(core_x))
            gate = self.activation_gate(self.gate_norm(gate_x))
        return core * gate # gate mul

    def forward_with_residual(
        self,
        feas: Tensor,
        residual: Tensor,
        res_weight: Tensor,
        *,
        dispatcher=None,
    ) -> Tensor | None:
        # The OpDispatcher's gated_ln_residual policy; None reads the module
        # switches (legacy dispatch). The dispatcher, not ints, is passed: an
        # int argument of this frame would be symbolic under torch.compile,
        # the dispatcher reads its threshold from a module global.
        if dispatcher is None:
            enabled, min_rows = _FUSED_GATED_LN_RESIDUAL, _LN_FUSE_THRESH
        else:
            enabled, min_rows = dispatcher.gated_ln_residual()
        if not enabled:
            rec = _recorder._ACTIVE
            if rec is not None:
                _record_residual_decline(rec, "disabled", residual)
            return None
        if not (
            self._fuse_ln
            and feas.is_cuda
            and residual.is_cuda
            and res_weight.is_cuda
            and residual.dim() == 2
            and residual.shape[0] >= min_rows
            and residual.shape[1] == res_weight.numel()
            and not res_weight.requires_grad
        ):
            rec = _recorder._ACTIVE
            if rec is not None:
                _record_residual_decline(
                    rec, _residual_decline_reason(self, feas, residual, res_weight, min_rows),
                    residual)
            return None
        cn, gn = self.core_norm, self.gate_norm
        if (
            cn.weight.requires_grad
            or cn.bias.requires_grad
            or gn.weight.requires_grad
            or gn.bias.requires_grad
        ):
            rec = _recorder._ACTIVE
            if rec is not None:
                _record_residual_decline(rec, "trainable_norm", residual)
            return None
        biases = self._folded_biases()
        if biases is None:
            core_x = self.mlp_core(feas)
            gate_x = self.mlp_gate(feas)
        else:
            core_x = self.mlp_core(feas, False)
            gate_x = self.mlp_gate(feas, False)
        if core_x.shape != residual.shape or gate_x.shape != residual.shape:
            rec = _recorder._ACTIVE
            if rec is not None:
                _record_residual_decline(rec, "shape_mismatch", residual)
            return None
        from fastmd.models.matris.op.triton_gated_ln import fused_gated_ln_residual

        return fused_gated_ln_residual(
            core_x,
            gate_x,
            cn.weight,
            cn.bias,
            gn.weight,
            gn.bias,
            residual,
            res_weight,
            cn.eps,
            gn.eps,
            *(() if biases is None else biases),
        )

    def _folded_biases(self, x: Tensor | None = None):
        """D13a: the second linears' (core, gate) biases when the fused gated LayerNorm adds
        them, else None (the linears add them). ``x``: the MLPs' input, qualified here for
        ``_finish_gate``'s fused gated LayerNorm (same guards); None: the caller qualified
        the fused kernel (``forward_with_residual``). Not inside torch.compile'd layers
        (enable_compile): there the row-threshold guard, moved ahead of the MLPs, added 7
        recompiles and 5 graph breaks to the RDX compile config (G-S3)."""
        if torch.compiler.is_compiling() or not _GATED_LN_BIAS_FOLD or self.use_fp16 or not self._fuse_ln:
            return None
        if x is not None:
            cn, gn = self.core_norm, self.gate_norm
            if not (x.is_cuda and x.shape[0] >= _LN_FUSE_THRESH) or (
                    cn.weight.requires_grad or cn.bias.requires_grad
                    or gn.weight.requires_grad or gn.bias.requires_grad):
                return None
        core_last, gate_last = self.mlp_core.layers[-1], self.mlp_gate.layers[-1]
        # Exactly nn.Linear: the fold calls F.linear with its weight instead of the module.
        if not (type(core_last) is nn.Linear and type(gate_last) is nn.Linear
                and core_last.bias is not None and gate_last.bias is not None):
            return None
        return core_last.bias, gate_last.bias

    def _first_linear_modules(self) -> tuple[nn.Linear, nn.Linear] | None:
        if len(self.mlp_core.layers) == 0 or len(self.mlp_gate.layers) == 0:
            return None
        core_linear = self.mlp_core.layers[0]
        gate_linear = self.mlp_gate.layers[0]
        if not isinstance(core_linear, nn.Linear) or not isinstance(gate_linear, nn.Linear):
            return None
        return core_linear, gate_linear

    def _can_fuse_indexed_cat_first_linear(
        self, rows: int, dim, parts: int, *, projection_config=None,
    ) -> bool:
        """``dim``: the width of every part, or the per-part widths (D8)."""
        enabled = (_INDEXED_CAT_LINEAR_FUSE if projection_config is None
                   else projection_config.enabled)
        min_rows = (_INDEXED_CAT_LINEAR_MIN_ROWS if projection_config is None
                    else projection_config.min_rows)
        if not enabled:
            return False
        if self.use_fp16 or self.gate_norm is None or not self._fuse_ln:
            return False
        if rows < min_rows:
            return False
        modules = self._first_linear_modules()
        if modules is None:
            return False
        core_linear, gate_linear = modules
        widths = _part_widths(dim, parts)
        input_dim = sum(widths)
        if (
            len(widths) != parts
            or core_linear.in_features != input_dim
            or gate_linear.in_features != input_dim
            or core_linear.out_features != gate_linear.out_features
        ):
            return False
        if not _mixed_width_allowed(widths, projection_config):
            return False
        if (core_linear.bias is None) != (gate_linear.bias is None):
            return False
        tensors = [core_linear.weight, gate_linear.weight]
        if core_linear.bias is not None:
            tensors += [core_linear.bias, gate_linear.bias]
        tensors += [self.core_norm.weight, self.core_norm.bias, self.gate_norm.weight, self.gate_norm.bias]
        return not any(t.requires_grad for t in tensors)

    def _split_first_linear_params(
        self,
        rows: int,
        dim,
        parts: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        projection_config=None,
    ) -> tuple[tuple[Tensor, ...], Tensor | None] | None:
        if not self._can_fuse_indexed_cat_first_linear(
            rows, dim, parts, projection_config=projection_config,
        ):
            return None
        modules = self._first_linear_modules()
        if modules is None:
            return None
        core_linear, gate_linear = modules
        if core_linear.weight.device != device or core_linear.weight.dtype != dtype:
            return None
        if gate_linear.weight.device != device or gate_linear.weight.dtype != dtype:
            return None
        bias_key = ()
        if core_linear.bias is not None:
            if core_linear.bias.device != device or core_linear.bias.dtype != dtype:
                return None
            if gate_linear.bias.device != device or gate_linear.bias.dtype != dtype:
                return None
            bias_key = (
                core_linear.bias.data_ptr(),
                gate_linear.bias.data_ptr(),
                core_linear.bias._version,
                gate_linear.bias._version,
            )
        key = (
            core_linear.weight.data_ptr(),
            gate_linear.weight.data_ptr(),
            core_linear.weight._version,
            gate_linear.weight._version,
            device,
            dtype,
            _part_widths(dim, parts),
            bias_key,
        )
        if self._indexed_cat_linear_split_cache_key != key:
            if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                rec = _recorder._ACTIVE
                if rec is not None:
                    rec.fallback("indexed_projection_weights", "split_cache", "cold_cache_under_capture",
                                 {"parts": parts, "width": _part_widths(dim, parts)[0]})
                return None
            with torch.no_grad():
                weights = []
                start = 0
                for width in _part_widths(dim, parts):
                    end = start + width
                    weights.append(
                        torch.cat(
                            [
                                core_linear.weight.detach()[:, start:end],
                                gate_linear.weight.detach()[:, start:end],
                            ],
                            dim=0,
                        ).contiguous()
                    )
                    start = end
                self._indexed_cat_linear_split_weights = tuple(weights)
                if core_linear.bias is None:
                    self._indexed_cat_linear_split_bias = None
                else:
                    self._indexed_cat_linear_split_bias = torch.cat(
                        [core_linear.bias.detach(), gate_linear.bias.detach()],
                        dim=0,
                    ).contiguous()
                self._indexed_cat_linear_split_cache_key = key
                from fastmd.engine.cache_provenance import publish_projection_cache
                sources = (core_linear.weight, gate_linear.weight)
                if core_linear.bias is not None:
                    sources += (core_linear.bias, gate_linear.bias)
                publish_projection_cache(self, "indexed_split", sources)
        if not torch.compiler.is_compiling() and _CAPTURE.holder is not None:  # a capture keeps what it reads
            sources = (core_linear.weight, gate_linear.weight)
            if core_linear.bias is not None:
                sources += (core_linear.bias, gate_linear.bias)
            retain_under_capture((self._indexed_cat_linear_split_weights,
                                  self._indexed_cat_linear_split_bias), sources)
        return self._indexed_cat_linear_split_weights, self._indexed_cat_linear_split_bias

    @staticmethod
    def _run_after_first(mlp: MLP, x: Tensor, fold: bool = False) -> Tensor:
        # fold (D13a): the last linear without its bias (the fused gated LayerNorm adds it).
        if fold:
            return _run_layers(list(mlp.layers)[1:], x, True)
        for layer in list(mlp.layers)[1:]:
            x = _apply_layer(layer, x)
        return x

    @staticmethod
    def _run_after_first_activation(mlp: MLP, x: Tensor, fold: bool = False) -> Tensor:
        if fold:
            return _run_layers(list(mlp.layers)[2:], x, True)
        for layer in list(mlp.layers)[2:]:
            x = _apply_layer(layer, x)
        return x

    def aligned_first_linear_block(
        self,
        rows: int,
        dim,
        parts: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        projection_config=None,
    ) -> Tensor | None:
        """Aligned-column block of the split first linear ([2H, width of the
        aligned part]) or None. ``dim`` as in ``_split_first_linear_params``.

        Lets a caller merge the p0 projection into a wider frozen GEMM and
        pass the result back via ``precomputed_p0``. Only valid for the
        decomposed backend, under the same fusion guards.
        """
        backend = (_INDEXED_CAT_LINEAR_BACKEND if projection_config is None
                   else projection_config.backend)
        if backend != "decomposed_indexed_silu":
            rec = _recorder._ACTIVE
            if rec is not None:
                rec.fallback("aligned_first_linear_block", backend, "backend_not_decomposed",
                             {"rows": rows, "width": _part_widths(dim, parts)[0], "parts": parts})
            return None
        params = self._split_first_linear_params(
            rows, dim, parts, device, dtype, projection_config=projection_config,
        )
        if params is None:
            rec = _recorder._ACTIVE
            if rec is not None:
                reason = _first_linear_decline_reason(
                    self, rows, dim, parts, device, dtype, projection_config)
                rec.fallback("aligned_first_linear_block", backend, "split_params:" + reason,
                             {"rows": rows, "width": _part_widths(dim, parts)[0], "parts": parts})
            return None
        weights, _ = params
        return weights[0]

    def forward_aligned_gather_cat3(
        self,
        aligned: Tensor,
        gathered: Tensor,
        index_a: Tensor,
        index_b: Tensor,
        index_a_sorted: bool = False,
        precomputed_p0: Tensor | None = None,
        *,
        projection_config=None,
    ) -> Tensor | None:
        if not (
            aligned.is_cuda
            and gathered.is_cuda
            and (aligned.is_contiguous() or precomputed_p0 is not None)
            and gathered.is_contiguous()
        ):
            rec = _recorder._ACTIVE
            if rec is not None:
                _record_indexed_decline(
                    rec, "indexed_cat3",
                    None if projection_config is None else projection_config.backend,
                    "tensor_guard", aligned)
            return None
        rows = aligned.shape[0]
        # The per-part widths (D8: they may differ, e.g. [32|128|128]); always a tuple from
        # the model's call sites, so the helpers see one argument type under torch.compile.
        dim = (aligned.shape[1], gathered.shape[1], gathered.shape[1])
        backend = (_INDEXED_CAT_LINEAR_BACKEND if projection_config is None
                   else projection_config.backend)
        if backend == "decomposed_indexed_silu":
            params = self._split_first_linear_params(
                rows, dim, 3, aligned.device, aligned.dtype,
                projection_config=projection_config,
            )
            if params is None:
                rec = _recorder._ACTIVE
                if rec is not None:
                    reason = _first_linear_decline_reason(
                        self, rows, dim, 3, aligned.device, aligned.dtype, projection_config)
                    _record_indexed_decline(
                        rec, "indexed_cat3", backend, "split_params:" + reason, aligned)
                return None
            weights, bias = params
            from fastmd.engine.kernels.decomposed_indexed_cat_silu_linear import decomposed_indexed_cat3_silu_linear

            fused_first = decomposed_indexed_cat3_silu_linear(
                aligned,
                gathered,
                index_a,
                index_b,
                weights,
                bias,
                index_a_sorted,
                precomputed_p0=precomputed_p0,
                strict_errors=(projection_config is not None and projection_config.strict_errors),
            )
            if fused_first is None:
                rec = _recorder._ACTIVE
                if rec is not None:
                    _record_indexed_decline(
                        rec, "indexed_cat3", backend, "kernel_declined", aligned)
                return None
            core_first, gate_first = fused_first
            out = self._gated_tail(core_first, gate_first)
            if out is not None:
                return out
            biases = self._folded_biases(core_first)
            core_x = self._run_after_first_activation(self.mlp_core, core_first, biases is not None)
            gate_x = self._run_after_first_activation(self.mlp_gate, gate_first, biases is not None)
            scope = self._register_gate_inputs(core_x, gate_x, 2, core_first, gate_first, biases is not None)
            out = self._finish_gate(core_x, gate_x, biases)
            self._register_gate_output(scope, out, core_x, gate_x, biases)
            return out

        rec = _recorder._ACTIVE
        if rec is not None:
            _record_indexed_decline(
                rec, "indexed_cat3", backend, "unsupported_backend", aligned)
        return None

    def forward_aligned_gather_cat4(
        self,
        aligned: Tensor,
        gathered_a: Tensor,
        index_a: Tensor,
        gathered_b: Tensor,
        index_b: Tensor,
        index_c: Tensor,
        index_a_sorted: bool = False,
        precomputed_p0: Tensor | None = None,
        *,
        projection_config=None,
    ) -> Tensor | None:
        if not (
            aligned.is_cuda
            and gathered_a.is_cuda
            and gathered_b.is_cuda
            and (aligned.is_contiguous() or precomputed_p0 is not None)
            and gathered_a.is_contiguous()
            and gathered_b.is_contiguous()
        ):
            rec = _recorder._ACTIVE
            if rec is not None:
                _record_indexed_decline(
                    rec, "indexed_cat4",
                    None if projection_config is None else projection_config.backend,
                    "tensor_guard", aligned)
            return None
        rows = aligned.shape[0]
        # The per-part widths (D8: they may differ, e.g. [32|128|32|32]); see cat3.
        dim = (aligned.shape[1], gathered_a.shape[1], gathered_b.shape[1], gathered_b.shape[1])
        backend = (_INDEXED_CAT_LINEAR_BACKEND if projection_config is None
                   else projection_config.backend)
        if backend == "decomposed_indexed_silu":
            params = self._split_first_linear_params(
                rows, dim, 4, aligned.device, aligned.dtype,
                projection_config=projection_config,
            )
            if params is None:
                rec = _recorder._ACTIVE
                if rec is not None:
                    reason = _first_linear_decline_reason(
                        self, rows, dim, 4, aligned.device, aligned.dtype, projection_config)
                    _record_indexed_decline(
                        rec, "indexed_cat4", backend, "split_params:" + reason, aligned)
                return None
            weights, bias = params
            from fastmd.engine.kernels.decomposed_indexed_cat_silu_linear import decomposed_indexed_cat4_silu_linear

            fused_first = decomposed_indexed_cat4_silu_linear(
                aligned,
                gathered_a,
                index_a,
                gathered_b,
                index_b,
                index_c,
                weights,
                bias,
                index_a_sorted,
                precomputed_p0=precomputed_p0,
                strict_errors=(projection_config is not None and projection_config.strict_errors),
            )
            if fused_first is None:
                rec = _recorder._ACTIVE
                if rec is not None:
                    _record_indexed_decline(
                        rec, "indexed_cat4", backend, "kernel_declined", aligned)
                return None
            core_first, gate_first = fused_first
            out = self._gated_tail(core_first, gate_first)
            if out is not None:
                return out
            biases = self._folded_biases(core_first)
            core_x = self._run_after_first_activation(self.mlp_core, core_first, biases is not None)
            gate_x = self._run_after_first_activation(self.mlp_gate, gate_first, biases is not None)
            scope = self._register_gate_inputs(core_x, gate_x, 2, core_first, gate_first, biases is not None)
            out = self._finish_gate(core_x, gate_x, biases)
            self._register_gate_output(scope, out, core_x, gate_x, biases)
            return out

        rec = _recorder._ACTIVE
        if rec is not None:
            _record_indexed_decline(
                rec, "indexed_cat4", backend, "unsupported_backend", aligned)
        return None

    def _second_layers(self, start: int, core_in: Tensor, gate_in: Tensor, fused: str | None = "tf32x3",
                       fold: bool = False):
        """Layers ``start:`` of both MLPs (a recompute recipe of the gate inputs).

        ``fused``: the one-kernel rebuild's dot precision, bound at
        registration (``triton_dot_input_precision("act_linear")``: 3xTF32
        by default where cuBLAS runs IEEE FP32); None: the forward's own
        cuBLAS layers rebuild the values (by default under TF32). ``fold``:
        without the last linears' biases, as the forward (D13a)."""
        linears = self._second_linears(start) if fused else None
        if linears is not None:
            from fastmd.models.matris.op.triton_act_linear import act_linear_pair
            from fastmd.models.matris.op.triton_act_linear import act_linear_pair_supported

            if (act_linear_pair_supported(core_in, linears[0].weight)
                    and act_linear_pair_supported(gate_in, linears[1].weight)):
                return act_linear_pair(core_in, gate_in, *linears, silu=start == 1, bias=not fold,
                                       precision=fused)
        return (_run_layers(list(self.mlp_core.layers)[start:], core_in, fold),
                _run_layers(list(self.mlp_gate.layers)[start:], gate_in, fold))

    def _gated_tail(self, core_h: Tensor, gate_h: Tensor) -> Tensor | None:
        """C2 (D13b): layers ``2:`` of both MLPs (Dropout(0), the second Linear) and ``_finish_gate`` in one
        kernel per direction (``triton_gated_tail``), from the first activations' outputs; None: the caller
        runs them unfused. Under the fused gated LayerNorm's guards, for frozen second linears within the
        kernel's widths; not inside torch.compile'd layers (their compiled program stays unchanged). In a
        saved-tensor recompute scope the kernel registers its own recipe, so every level runs the
        forward of the off mode."""
        if torch.compiler.is_compiling() or not _FUSED_GATED_TAIL or self.use_fp16 or not self._fuse_ln:
            return None
        if not (core_h.is_cuda and core_h.shape[0] >= _LN_FUSE_THRESH):
            return None
        from fastmd.models.matris.op.triton_gated_tail import fused_gated_tail
        from fastmd.models.matris.op.triton_gated_tail import gated_tail_supported

        cn, gn = self.core_norm, self.gate_norm
        linears = self._second_linears(2)
        if linears is None or type(linears[0]) is not nn.Linear or type(linears[1]) is not nn.Linear:
            reason = "layers"
        elif (linears[0].bias is None) != (linears[1].bias is None):
            reason = "bias_mismatch"
        elif any(t is not None and t.requires_grad for t in (
                cn.weight, cn.bias, gn.weight, gn.bias, linears[0].weight, linears[0].bias,
                linears[1].weight, linears[1].bias)):
            reason = "trainable_params"
        elif not gated_tail_supported(core_h, gate_h, linears[0].weight, linears[1].weight):
            reason = "unsupported_shape"
        elif triton_dot_input_precision("gated_tail") is None:
            reason = "cublas_precision"
        elif not _saved_recompute.gated_tail_allowed():
            reason = "memory"  # the memory admission's decision (saved_recompute.gated_tail_allowed)
        else:
            core_linear, gate_linear = linears
            # The Cat3/Cat4 activations feed only this tail (their VJPs save the preact): overwrite them.
            return fused_gated_tail(core_h, gate_h, core_linear.weight, core_linear.bias, gate_linear.weight,
                                    gate_linear.bias, cn.weight, cn.bias, gn.weight, gn.bias, cn.eps, gn.eps,
                                    inplace=True)
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.fallback("gated_tail", "triton", reason, {"rows": core_h.shape[0], "width": core_h.shape[-1]})
        return None

    def _second_linears(self, start: int):
        """The second linears when layers ``start:`` are (SiLU,) Dropout(0), Linear."""
        pair = []
        for mlp in (self.mlp_core, self.mlp_gate):
            layers = list(mlp.layers)
            if len(layers) != 4 or not isinstance(layers[3], nn.Linear):
                return None
            if not isinstance(layers[1], (FusedSiLU, nn.SiLU)):
                return None
            if not isinstance(layers[2], nn.Dropout) or (layers[2].p and layers[2].training):
                return None
            pair.append(layers[3])
        return tuple(pair) if start in (1, 2) else None

    def _register_gate_inputs(self, core_x, gate_x, start, core_in, gate_in, fold=False):
        # Saved-tensor recompute: the gate's inputs are rebuilt from the first-layer outputs (saved
        # anyway, or themselves rebuilt); registered before the gate saves them. fold: bias-free (D13a).
        scope = _saved_recompute.active()
        if scope is not None:
            core_ref, gate_ref = scope.ref(core_in), scope.ref(gate_in)
            preact = _saved_recompute.silu_halves_source(core_ref, gate_ref)
            fused = triton_dot_input_precision("act_linear")
            if fused and preact is not None and self._second_linears(1) is not None:
                # Cat3/Cat4 activations: start from their saved preact instead.
                half = preact.shape[1] // 2
                scope.register((core_x, gate_x), self._second_layers, 1, preact[:, :half], preact[:, half:],
                               fused, fold)
            else:
                scope.register((core_x, gate_x), self._second_layers, start, core_ref, gate_ref, fused, fold)
        return scope

    def _register_gate_output(self, scope, out, core_x, gate_x, biases=None) -> None:
        if scope is not None:
            scope.register((out,), self._finish_gate, scope.ref(core_x), scope.ref(gate_x), biases)

    def forward_gathered_chunks(self, aligned: Tensor, gathered, chunk: int) -> Tensor:
        """Saved-tensor recompute: ``forward(cat([aligned, value[index], ...], 1))`` with the
        concatenation built and projected by the (frozen) first linears
        ``chunk`` rows at a time, so that it never exists whole; the rest as in
        ``forward``."""
        from fastmd.models.matris.op.chunked_gather_linear import chunked_gather_linear

        core_linear, gate_linear = self.mlp_core.layers[0], self.mlp_gate.layers[0]
        core_first, gate_first = chunked_gather_linear(
            aligned, gathered,
            ((core_linear.weight, core_linear.bias), (gate_linear.weight, gate_linear.bias)), chunk)
        biases = self._folded_biases(aligned)
        core_x = self._run_after_first(self.mlp_core, core_first, biases is not None)
        gate_x = self._run_after_first(self.mlp_gate, gate_first, biases is not None)
        scope = self._register_gate_inputs(core_x, gate_x, 1, core_first, gate_first, biases is not None)
        out = self._finish_gate(core_x, gate_x, biases)
        self._register_gate_output(scope, out, core_x, gate_x, biases)
        return out

    def forward(self, feas: Tensor) -> Tensor:
        """
        Args:
            feas (Tensor): shape (feas_num, input_dim)
        Returns:
            output: shape (feas_num, output_dim)
        """
        if (_saved_recompute.active() is not None and not self.use_fp16
                and self._first_linear_modules() is not None):
            # Saved-tensor recompute: keep the first linears' outputs (the activations save them).
            precisions = (_layer_precision(self.mlp_core.layers[0], feas),
                          _layer_precision(self.mlp_gate.layers[0], feas))
            core_first = _apply_layer(self.mlp_core.layers[0], feas, precisions[0])
            gate_first = _apply_layer(self.mlp_gate.layers[0], feas, precisions[1])
            scope = _saved_recompute.active()
            if scope.mode == "deep":
                # deep: rebuild them too when their input is itself rebuilt.
                source = scope.ref(feas)
                parts = _saved_recompute.gather_cat_parts(source)
                for layer, first, precision in ((self.mlp_core.layers[0], core_first, precisions[0]),
                                                (self.mlp_gate.layers[0], gate_first, precisions[1])):
                    if parts is not None:
                        scope.register((first,), _first_linear_by_parts, *parts, layer.weight, layer.bias)
                    elif source is not feas:
                        scope.register((first,), _apply_layer, layer, source, precision)
            biases = self._folded_biases(feas)
            core_x = self._run_after_first(self.mlp_core, core_first, biases is not None)
            gate_x = self._run_after_first(self.mlp_gate, gate_first, biases is not None)
            scope = self._register_gate_inputs(core_x, gate_x, 1, core_first, gate_first, biases is not None)
            out = self._finish_gate(core_x, gate_x, biases)
            self._register_gate_output(scope, out, core_x, gate_x, biases)
            return out
        biases = self._folded_biases(feas)
        if biases is not None:
            return self._finish_gate(self.mlp_core(feas, False), self.mlp_gate(feas, False), biases)
        core_x = self.mlp_core(feas)
        gate_x = self.mlp_gate(feas)
        return self._finish_gate(core_x, gate_x)


class MOE_Layer(nn.Module):
    
    def __init__(
        self,
        num_expert: int = 64,
        input_dim: int = 128,
        hidden_dim: int | Sequence[int] | None = (128, 128),
        output_dim: int = 128,
        dropout: float = 0.0,
        activation: Literal["silu", "relu", "tanh", "gelu"] = "silu",
        bias: bool = True,
        use_fp16: bool = False,
    ):
        """Initialize the MOE layer.

        Args:
            
        """
        super().__init__()
        
        raise NotImplementedError
         
    def forward(self, feas: Tensor) -> Tensor:
        return None


class GraphPooling(nn.Module):
    def __init__(self, average: bool = False) -> None:
        
        super().__init__()
        self.average = average

    def forward(self, node_feat: Tensor, segment: Tensor, num_segment: int | None = None) -> Tensor:
        """
        Args:
            atom_feat (Tensor): batched atom features after convolution layers.
                [num_batch_atoms, node_feat_dim or 1]
            segment (Tensor): graph indices for each atom.
                [num_batch_atoms]
            num_segment (int): number of output segments (e.g. n_graphs). When
                provided, avoids a host max()-read (D2H sync).

        Returns:
            crystal_feas (Tensor): crystal feature matrix.
                [n_crystals, node_feat_dim or 1]
        """
        if self.average:
            if num_segment is not None:
                bin_count = segment_count(segment, num_segment)  # sync-free
            else:
                bin_count = torch.bincount(segment)
            bin_count = bin_count.where(bin_count != 0, bin_count.new_ones(1))
            n_seg = bin_count.shape[0]
        else:
            # average=False only needs the output size; avoid the host max()-read.
            n_seg = num_segment if num_segment is not None else int(segment.max()) + 1

        output = node_feat.new_zeros([n_seg, node_feat.shape[1]])
        output = output.index_add_(0, segment, node_feat)
        if self.average:
            output = (output.T / bin_count).T
        return output


def cg_change_mat(ang_mom: int, device: str = "cpu") -> torch.tensor:
    if ang_mom not in [2]:
        raise NotImplementedError

    if ang_mom == 2:
        change_mat = torch.tensor(
            [
                [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
                [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
                [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
                [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
                [0, 0, 0.5**0.5, 0, 0, 0, 0.5**0.5, 0, 0],
                [0, 2 ** (-0.5), 0, 2 ** (-0.5), 0, 0, 0, 0, 0],
                [
                    -(6 ** (-0.5)),
                    0,
                    0,
                    0,
                    2 * 6 ** (-0.5),
                    0,
                    0,
                    0,
                    -(6 ** (-0.5)),
                ],
                [0, 0, 0, 0, 0, 2 ** (-0.5), 0, 2 ** (-0.5), 0],
                [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
            ],
            device=device,
        ).detach()

    return change_mat


def irreps_sum(ang_mom: int) -> int:
    """
    Returns the sum of the dimensions of the irreps up to the specified angular momentum.

    :param ang_mom: max angular momenttum to sum up dimensions of irreps
    """
    total = 0
    for i in range(ang_mom + 1):
        total += 2 * i + 1

    return total


def reshape_stress(L0out, L2out, batch_size=1):
    _max_rank = 2
    pred_irreps = torch.zeros(
        (batch_size, irreps_sum(_max_rank)),
        device = L0out.device,
    )
    # L=0
    L=0
    pred_irreps[: ,irreps_sum(L-1): irreps_sum(L)] = L0out.view(batch_size, -1)
    
    L=2
    pred_irreps[: ,irreps_sum(L-1): irreps_sum(L)] = L2out.view(batch_size, -1) 
    
    pred = torch.einsum(
        "ba, cb->ca",
        cg_change_mat(_max_rank, device = L0out.device),
        pred_irreps,
    )
    
    return pred.view(batch_size, 3,3)


class Sphere(nn.Module):
    
    def __init__(self, lmax=2):
        super(Sphere, self).__init__()
        self.lmax = lmax
        
    def forward(self, edge_vec):
        edge_sh = self._spherical_harmonics(self.lmax, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2])
        return edge_sh
        
    @staticmethod
    def _spherical_harmonics(lmax: int, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        sh_0_0 = torch.ones_like(x)
        if lmax == 0:
            return torch.stack([ sh_0_0, ], dim=-1)
        
        sh_1_0, sh_1_1, sh_1_2 = x, y, z
        
        if lmax == 1:
            return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2], dim=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4], dim=-1)
