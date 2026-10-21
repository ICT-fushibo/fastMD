"""MatRIS's kernels. ``fused_silu`` and ``fused_sigmoid`` (the activation kernels), and a module accessed as an
attribute of the package, are imported on first access: importing one kernel module does not import the others."""
from fastmd import _lazy

__all__ = [
    "fused_silu",
    "fused_sigmoid",
]
__getattr__, __dir__ = _lazy.lazy_attributes(__name__, {"fused_silu": "fastmd.models.matris.op.fused_silu_op",
                                                        "fused_sigmoid": "fastmd.models.matris.op.fuse_sigmoid_op"})
