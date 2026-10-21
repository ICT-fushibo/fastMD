"""Communication interface used by local integrators and output writers.

FastMD provides LocalCommunicator: reductions and broadcasts are local
operations, and gather_by_id orders this simulation's atoms. No distributed
transport implementation is included."""
from __future__ import annotations

from numbers import Integral
from typing import Any, Callable, NoReturn, Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor

NOT_YET = "declared by the 3.1a Communicator protocol; implemented in a later unified-engine step"


def gather_fields(ids: Tensor, fields: dict) -> str | None:
    """This rank's ``gather_by_id`` input error, or None: 1-D integer IDs, row-aligned fields."""
    if (not isinstance(ids, Tensor) or ids.ndim != 1 or ids.is_floating_point()
            or ids.is_complex() or ids.dtype == torch.bool):
        return "gather_by_id: ids must be a one-dimensional integer tensor"
    if "ids" in fields:
        return "gather_by_id: 'ids' is reserved for the stable IDs"
    for name, value in fields.items():
        if (not isinstance(value, Tensor) or value.ndim == 0 or value.shape[0] != ids.shape[0]
                or value.is_complex()):
            return f"gather_by_id: field {name!r} must be a real tensor with leading dimension n_local"
    return None


def flag_parts(flags: Tensor | None, host: Sequence[int]) -> tuple[Tensor | None, tuple[int, ...]]:
    """Admit one flag round's inputs: device/CPU integer flags plus host integers.

    ``flags`` is ``None`` or a zero/one-dimensional bool or integer tensor (a
    device predicate stays where it was computed: the round copies it, never
    downloads it separately). ``host`` holds values the host already knows
    (an error bit, an epoch); they precede ``flags`` in the agreed vector.
    Every rank must pass the same number of entries in every round.
    """
    if flags is not None:
        if (not isinstance(flags, Tensor) or flags.layout != torch.strided or flags.ndim > 1
                or flags.is_floating_point() or flags.is_complex()):
            raise TypeError("flags must be a zero/one-dimensional bool or integer tensor")
        flags = flags.reshape(-1)
    values = tuple(host)
    if any(not isinstance(value, Integral) for value in values):
        raise TypeError("host flag entries must be integers")
    values = tuple(int(value) for value in values)
    if any(not -(1 << 62) <= value < (1 << 62) for value in values):
        # Room for the MAX-of-negation idiom (max(-x) = -min(x)) in int64.
        raise ValueError("host flag entries must fit in 63-bit signed integers")
    return flags, values


@runtime_checkable
class Communicator(Protocol):
    """Rank-consistent agreement and fixed-shape step communication (§6.2).

    Every rank calls every member in the same order. Epoch hooks run outside
    any CUDA graph on committed state; step hooks have fixed shapes, are
    stream ordered and may be captured. ``agree_*``/``check_config`` return
    (or raise) identically on every rank: all decisions derive from the same
    reduced/gathered values.
    """

    rank: int
    world_size: int

    # --- epoch hooks: outside any CUDA graph, only on committed state ---
    def begin_epoch(self, owned, per_atom_state, *, halo_cutoff: float):
        """Migrate plus halo(rc + skin); per-peer ghost slices and send tables (step 3.4)."""
        ...

    def agree_capacity(self, demand: Any, policy: Callable[[list, Any], Any],
                       current: Any) -> tuple[Any, bool]:
        """``(key, recapture_any)``: ``policy(demands_by_rank, current)`` on every rank's demand.

        ``policy`` must be a pure function of its (replicated) arguments, so
        every rank computes the same key; ``current`` must agree across ranks.
        """
        ...

    def agree_flags_host(self, flags: Tensor | None = None, *, phase: str,
                         host: Sequence[int] = ()) -> Tensor:
        """MAX over ranks of ``[*host, *flags]``; returns a CPU int64 vector.

        One reduction and one download per round. ``phase`` names the round;
        a round whose phase, length or order differs across ranks raises on
        every rank instead of silently reducing unrelated vectors.
        """
        ...

    def check_config(self, fingerprint: str) -> None:
        """Raise on every rank unless every rank's dispatch fingerprint is equal."""
        ...

    def gather_by_id(self, root: int, ids: Tensor, **fields) -> dict | None:
        """Per-atom rows of every rank on ``root``, ordered by stable atom ID (log/restart steps).

        ``fields`` have leading dimension n_local (float or integer dtypes;
        integers below 2**53). ``root`` receives ``{"ids": ..., **fields}``
        with ascending IDs and the original dtypes on the communicator's
        device; other ranks receive None. Not a step hook: it synchronizes.
        """
        ...

    def atom_copy_binding(self, name: str):
        """Epoch-frozen atom ghost COPY binding for ExchangeCopy (§5.2)."""
        ...

    def row_route_binding(self, name: str):
        """Resident per-peer-cap row route binding; indices recomputed in-graph (§5.2)."""
        ...

    def slice_exchange(self, send_rows: Tensor, send_counts: Sequence[int],
                       receive_counts: Sequence[int], *, transport: str = "neighbor_p2p"):
        """Epoch-frozen per-peer contiguous slices with ``copy(owned)`` / ``reverse_sum(cotangent, n)``.

        The generic vertex-level exchange (A1): owned rows ``send_rows``,
        grouped by peer, go to each peer's receive slice (COPY); the adjoint
        sends receive-shaped cotangents back and index-adds them into owned
        rows (reverse-SUM). One grouped P2P per call over the peers with rows.
        """
        ...

    def fatal(self, exc: BaseException) -> NoReturn:
        """K2: report, abort process groups within a bound, exit. Never returns."""
        ...

    # --- step hooks: fixed shape, stream ordered, capturable ---
    def ghost_copy_(self, rows: Tensor) -> None:
        """Owned coordinates into the ghost slices of ``pos_local`` (step 3.4)."""
        ...

    def ghost_reverse_sum_(self, rows: Tensor) -> Tensor:
        """Adjoint of ``ghost_copy_``: deterministic index_add_ into owned rows (step 3.4)."""
        ...

    def allreduce_sum_(self, vec: Tensor) -> None:
        """In-place SUM of a fused vector ``[E_local, 2*KE, sum(p) (3), ...]``; no host sync."""
        ...

    def allreduce_max_(self, flags: Tensor) -> None:
        """In-place MAX of an int64 flag vector; no host sync (read later via one download)."""
        ...


class LocalCommunicator:
    """P=1: agreements are the local values, step hooks no-ops, no process groups.

    ``agree_flags_host`` is the one download of the local vector (the P>1
    implementation's reduction degenerates to it), so a caller written
    against the protocol pays the same single D2H at P=1.
    """

    rank = 0
    world_size = 1

    def agree_capacity(self, demand, policy, current):
        key = policy([demand], current)
        return key, key != current

    def agree_flags_host(self, flags=None, *, phase, host=()):
        if not isinstance(phase, str):
            raise TypeError("flag round phase must be a string")
        flags, values = flag_parts(flags, host)
        local = torch.tensor(values, dtype=torch.long)
        if flags is None or not flags.numel():
            return local
        return torch.cat((local, flags.to(device="cpu", dtype=torch.long)))

    def check_config(self, fingerprint):
        if not isinstance(fingerprint, str):
            raise TypeError("configuration fingerprint must be a string")

    def errors_slow_path(self, error, flag_any, *, phase="errors"):
        """The (single) rank's error list; see DomainCommunicator.errors_slow_path."""
        return [error]

    def fatal(self, exc):
        # No peers to release: the caller's normal error path owns the exit.
        raise exc

    def allreduce_sum_(self, vec):
        return None

    def allreduce_max_(self, flags):
        return None

    def slice_exchange(self, send_rows, send_counts, receive_counts, *, transport="neighbor_p2p"):
        if len(send_counts) != 1 or tuple(send_counts) != tuple(receive_counts):
            raise ValueError("one rank sends its only slice to itself")
        return _LocalSlices(send_rows)

    def gather_by_id(self, root, ids, **fields):
        """The local rows sorted by ID (``root`` must be 0)."""
        if root != 0:
            raise ValueError("gather_by_id: the only rank is 0")
        error = gather_fields(ids, fields)
        if error is not None:
            raise ValueError(error)
        order = torch.argsort(ids, stable=True)
        return {"ids": ids[order], **{name: value[order] for name, value in fields.items()}}

    def begin_epoch(self, owned, per_atom_state, *, halo_cutoff):
        raise NotImplementedError(NOT_YET)

    def atom_copy_binding(self, name):
        raise NotImplementedError(NOT_YET)

    def row_route_binding(self, name):
        raise NotImplementedError(NOT_YET)

    def ghost_copy_(self, rows):
        raise NotImplementedError(NOT_YET)

    def ghost_reverse_sum_(self, rows):
        raise NotImplementedError(NOT_YET)


class _LocalSlices:
    """P=1 slice exchange: the self slice is a gather and its adjoint an index_add_."""

    def __init__(self, send_rows: Tensor):
        self.send_rows = send_rows

    def copy(self, owned: Tensor) -> Tensor:
        return owned.index_select(0, self.send_rows.to(owned.device))

    def reverse_sum(self, cotangent: Tensor, owned_rows: int) -> Tensor:
        output = cotangent.new_zeros((owned_rows, *cotangent.shape[1:]))
        return output.index_add_(0, self.send_rows.to(cotangent.device), cotangent)
