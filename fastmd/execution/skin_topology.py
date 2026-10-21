"""Skin-epoch topology reuse for single-GPU fixed-frame evaluation (step C7).

``MLIPCalculator`` otherwise rebuilds the ragged graph every call
(``atoms_to_graph_gpu``: host synchronizations while the GPU idles). Here one
system keeps a candidate list with cutoff rc + skin
(``FixedCapacityGraphBuilder.refresh_candidates``) and every call filters it
exactly on the device (d < rc, d < r3, triplets enumerated per call) into
padded static buffers (the adapter's ``PaddingRule``; MatRIS: sink atoms),
followed by the model; with model_graph both run as
one captured CUDA graph per capacity bucket. A model that reads edges only
(``TopologySpec.triplet_cutoff`` None) gets the builders' edges-only mode: the
edges (d < rc), padded by the adapter's ``PaddingRule``, and the model's call
over ``adapter.prepare`` of the builder's ``GraphBatch``. A call is: copy the
positions in, one launch (or the eager body), one readback of the results and
a status vector.

Exactness: the candidate list is the only topology state carried between
calls. The host refreshes it before the launch once an atom moved
PROACTIVE_REFRESH_FRACTION of the skin since the last refresh (minimum image,
so ASE's ``atoms.wrap()`` is no motion), and the device checks every call that
no atom moved half the skin (``2 * maxdisp <= skin``: every pair within rc is
then a candidate). A violation or a capacity overflow (the builder then
evaluates a safe all-padding topology) discards the evaluation, which is redone
after a refresh and/or in a larger bucket; a truncated candidate row
(``SkinTopologyUnavailable``) sends the system back to the per-call rebuild.
Positions, atomic numbers (``set_atomic_numbers``) and every derived graph
tensor are device inputs rewritten per call; the cell and the periodicity are
fixed per runner. The padding is the adapter's padding rule, and the capacity
buckets, pool release and precapture are ``BucketedGraphRunner``'s (task O
policy); the demand that selects a bucket
is the previous call's device counts plus the ``min_pad_u`` / ``min_pad_t``
margins, or an exact count on refresh. Edges-only buckets keep the key's
triplet dimension (a constant: the demand has no triplets) and build no
triplets (``t_capacity`` 0).

``MATRIS_TOPOLOGY_REUSE_CHECK=N`` checks the topology of the first call of a
runner, every refresh, capture, growth or redo call and every N-th call
against (a) a builder with candidates refreshed at the same host positions
(bitwise) and (b) the per-call ragged build of the ASE atoms (the
topology_reuse="off" path): atomic numbers, lattice and fractional
coordinates bitwise, and the same edges and triplets up to pairs within
CUTOFF_TIE of a cutoff (the two builders test d^2 < rc^2 in a kernel vs
sqrt(sum) < rc in torch). For an edges-only model the per-call build is
``atoms_to_edges_gpu`` and the check covers its edges and wrapped positions (no
pairs or triplets). Energies and forces are compared by the gate harnesses,
not here (the reference evaluation would need a second pool or an eager
working set next to the captured one).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch

from fastmd._nvtx import nvtx_range
from fastmd.config import env_value
from fastmd.engine.policy import fusion_scope
from fastmd.graph.fixed_capacity_builder import FixedCapacityGraphBuilder
from fastmd.graph.fixed_capacity_builder import fixed_capacity_exceeded
from fastmd.graph.geometry import small_matmul
from fastmd.graph.gpu_graph_builder import atoms_to_edges_gpu
from fastmd.graph.gpu_graph_builder import atoms_to_graph_gpu
from fastmd.graph.validation import raise_if_isolated_atoms
from fastmd.engine.entry_config import scoped_call
from fastmd.engine._graph_capture import capture_graph
from fastmd.execution.cuda_graph import BucketedGraphRunner
from fastmd.execution.cuda_graph import _cap_for

# The host refreshes the candidates, before the launch, once an atom moved this
# fraction of the skin since the last refresh (0.9 of the device limit
# skin / 2: the host tests the exact positions it launches, so the margin only
# covers its float64 vs the device's float32 arithmetic).
PROACTIVE_REFRESH_FRACTION = 0.45
# After a cell change of the same system the calculator rebuilds per call
# until the cell has been unchanged for this many calls (model_graph: 4x per
# cell-change episode up to STABLE_CELL_CALLS_MAX, so a cell relaxation does
# not recapture every few calls).
STABLE_CELL_CALLS = 4
STABLE_CELL_CALLS_EAGER = 2
STABLE_CELL_CALLS_MAX = 256
# A pair this close to a cutoff (A) may be decided differently by the ragged
# builder (sqrt of a torch sum < rc) and the fixed builder (d^2 < rc^2).
CUTOFF_TIE = 2e-5
# Evaluations of one call: the first, a refresh, a growth, and one spare.
_MAX_ATTEMPTS = 4


class SkinTopologyUnavailable(RuntimeError):
    """The skin path cannot evaluate this system (the caller rebuilds per call)."""


class SkinTopologyRunner(BucketedGraphRunner):
    """Fixed-frame E/F(/S/M) of one periodic system with a skin-epoch topology.

    ``capture=False`` runs the same body eagerly (a builder per bucket, no
    graphs). ``evaluate(positions)`` takes host coordinates (float64, any
    lattice image) and returns host results. ``model``: a MatRIS module or
    an adapter (an edges-only one: builders without triplets).
    """

    def __init__(self, model, *, task, atomic_numbers, cell, pbc, skin, capture,
                 check_isolated_atoms, handle_isolated_atoms, device, enable_model_fusions=True,
                 composition=None, config=None, max_neighbor_count=None):
        self.device = torch.device(device)
        with torch.cuda.device(self.device):
            self._init(model, task=task, atomic_numbers=atomic_numbers, cell=cell, pbc=pbc, skin=skin,
                       capture=capture, check_isolated_atoms=check_isolated_atoms,
                       handle_isolated_atoms=handle_isolated_atoms, enable_model_fusions=enable_model_fusions,
                       composition=composition, config=config, max_neighbor_count=max_neighbor_count)

    def _init(self, model, *, task, atomic_numbers, cell, pbc, skin, capture, check_isolated_atoms,
              handle_isolated_atoms, enable_model_fusions, composition, config, max_neighbor_count=None):
        super().__init__(model, task=task, enable_model_fusions=enable_model_fusions, config=config)
        if not (np.isfinite(skin) and skin > 0):
            raise ValueError("topology reuse needs a positive capacity.candidate_skin")
        self.capture = bool(capture)
        self.skin = float(skin)
        self.check_isolated_atoms = bool(check_isolated_atoms)
        self.handle_isolated_atoms = bool(handle_isolated_atoms)
        self.n_real = int(len(atomic_numbers))
        self._n_real = self.n_real
        self.cell = np.array(cell, dtype=np.float64)
        self._inv_cell = np.linalg.inv(self.cell)
        self.pbc = np.array(pbc, dtype=bool)
        topology = self.adapter.topology_spec()
        self.atom_graph_cutoff = float(topology.cutoff)
        # A model without triplets reads edges only: builders without a line graph.
        self._edges_only = topology.triplet_cutoff is None
        self.line_graph_cutoff = None if self._edges_only else float(topology.triplet_cutoff)
        # The runner owns these device tensors (allocated once): the float32
        # cell and numbers exactly as atoms_to_graph_gpu makes them.
        self._numbers = np.array(atomic_numbers, dtype=np.int64)
        self._builder_args = dict(
            cell=torch.from_numpy(self.cell.astype(np.float32)).to(self.device),
            atomic_numbers=torch.tensor(self._numbers, dtype=torch.int32, device=self.device),
            pbc=torch.tensor(self.pbc, device=self.device),
            atom_graph_cutoff=self.atom_graph_cutoff,
            line_graph_cutoff=self.line_graph_cutoff,
            n_dummy=self.n_dummy,
            device=self.device,
            composition=composition,
            candidate_cutoff=self.atom_graph_cutoff + self.skin,
            minimum_image=True,
            # The padded rows the model needs and its own input over the
            # builder's batch (MatRIS: sink atoms and its RadiusGraph view,
            # the builder's defaults).
            padding=self.adapter.padding_rule(),
            prepare=self.adapter.prepare,
        )
        if self._derivative is not None and not self._edges_only:
            # A line graph the engine differentiates: the batch's positions are its leaf.
            self._builder_args["publish_positions"] = True
        if max_neighbor_count is not None:
            # A model with a neighbour cap (edges only): every build folds its largest per-atom edge count into
            # this [1] device maximum (the caller reads it).
            self._builder_args["max_neighbor_count"] = max_neighbor_count
        # Owns the candidate list every bucket's builder reads; its own
        # capacity is irrelevant (it only counts the demand).
        self.owner = FixedCapacityGraphBuilder(u_capacity=1, t_capacity=self._t_capacity(1), **self._builder_args)
        self.positions = torch.zeros((self.n_real, 3), dtype=torch.float32, device=self.device)
        self._pinned_positions = torch.empty((self.n_real, 3), dtype=torch.float32, pin_memory=True)
        self._pinned_flags = torch.empty(6, dtype=torch.int64, pin_memory=True)
        self._pinned_scalars = torch.empty(11, dtype=torch.float32, pin_memory=True)
        self._pinned_scalars64 = None  # FP64 energies (allocated at the first)
        self._pinned_forces = torch.empty((self.n_real, 3), dtype=torch.float32, pin_memory=True)
        self._pinned_magmoms = torch.empty(self.n_real, dtype=torch.float32, pin_memory=True)
        self._reference = None      # host coordinates of the last refresh
        self._demand = None         # (u, t) of the last evaluation
        self._event = False         # a refresh, capture, Z change or redo since the last check
        self.check_every = int(env_value("MATRIS_TOPOLOGY_REUSE_CHECK"))
        self._scratch = None        # checker (a): builder with its own candidates
        self.calls = 0
        self.refreshes = 0
        self.redo_capacity = 0
        self.redo_skin = 0
        self.checks = 0
        self.check_ties = 0
        self.max_host_displacement = 0.0
        self.last_edge_count = self.last_triplet_count = 0

    # --- buckets -----------------------------------------------------------

    def _t_capacity(self, t_cap):
        """A builder's triplet capacity for a bucket's triplet key: none for an edges-only model."""
        return 0 if self._edges_only else t_cap

    def _body(self, builder):
        graph, status = builder.build(self.positions, rebuild_candidates=False)
        if self._derivative is None:
            out = self.adapter.evaluate([graph], task=self.task, is_training=False, n_real=self.n_real,
                                        handle_isolated_atoms=self.handle_isolated_atoms)
        else:
            out = self._derivative(builder.batch, graph, task=self.task, n_real=self.n_real)
        flags = torch.cat([
            status.candidate_valid.long(),
            status.candidate_capacity_valid.long(),
            fixed_capacity_exceeded(status, builder).reshape(1).long(),
            status.edge_count.long(),
            status.triplet_count.long(),
            status.isolated_atom_count.long(),
        ])
        return out, flags

    @scoped_call
    def _capture(self, u_cap, t_cap, sample_g=None):
        builder = FixedCapacityGraphBuilder(
            u_capacity=u_cap, t_capacity=self._t_capacity(t_cap), candidates_from=self.owner, **self._builder_args)
        entry = dict(builder=builder, graph=None, out=None, flags=None, n_real=self.n_real)
        if self.capture:
            with fusion_scope(self.enable_model_fusions):
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    self._warm(lambda: self._body(builder))
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                if self._capture_stream is None:
                    self._capture_stream = torch.cuda.Stream()
                with capture_graph(graph, pool=self.pool, stream=self._capture_stream):
                    out, flags = self._body(builder)
            entry.update(graph=graph, out=out, flags=flags)
            self.captures += 1
        self.cache[(u_cap, t_cap)] = entry
        self._event = True

    def _release_pool(self):
        self._scratch = None
        if self.capture:
            super()._release_pool()
            return
        self.cache.clear()
        self.pool_resets += 1
        self._window = [0, 0, 0]

    def close(self) -> None:
        """Release every bucket (graphs and pool) of this system."""
        if self.cache:
            self._release_pool()
        self._tight.clear()
        self._envelope = None

    def run(self, g):
        raise NotImplementedError("SkinTopologyRunner evaluates positions (evaluate), not ragged graphs")

    def ensure(self, sample_g, key):
        """Capture bucket ``key`` (``sample_g`` unused: the body builds its own graph)."""
        if key not in self.cache:
            self._capture(*key)

    def _bucket_entry(self, n, u, t, sample_g):
        if self.capture:
            return super()._bucket_entry(n, u, t, sample_g)
        # Eager: one builder at the tight bucket of the demand. Padded rows
        # cost eager time and there is no capture to amortize, so no growth
        # headroom; it is replaced when the demand leaves it or it pads more
        # than one step beyond the tight bucket.
        key = _cap_for(u, t, self.u_step, self.t_step, self.min_pad_u)
        current = next(iter(self.cache), None)
        if (current is not None and current[0] > u and current[1] >= t
                and current[0] - key[0] <= self.u_step and current[1] - key[1] <= self.t_step):
            return self.cache[current]
        self._release_pool()
        self._capture(*key)
        return self.cache[key]

    def captures_next(self, g):
        raise NotImplementedError("SkinTopologyRunner evaluates positions, not ragged graphs")

    @scoped_call
    def precapture(self, proactive: bool = True) -> None:
        """BucketedGraphRunner.precapture at the last evaluation's demand."""
        if self._demand is None or not self.capture:
            return
        u, t = self._margin(*self._demand)
        # Edges only: a graph batch as the sample (the owner's; the body builds
        # its own) keeps the triplet key fixed, so no neighbor bucket in it.
        sample = self.owner.batch if self._edges_only else None
        with torch.cuda.device(self.device):
            self._precapture(self.n_real, *self._held_key(u, t, _cap_for(u, t, self.u_step, self.t_step,
                                                                          self.min_pad_u)),
                             sample, proactive)

    # --- evaluation --------------------------------------------------------

    def set_atomic_numbers(self, atomic_numbers) -> None:
        """New atomic numbers of the same atoms (in place, no refresh or capture)."""
        numbers = np.asarray(atomic_numbers, dtype=np.int64)
        if np.array_equal(numbers, self._numbers):
            return
        device_numbers = torch.tensor(numbers, dtype=torch.int32, device=self.device)
        if self._edges_only or self._derivative is not None:
            # A model whose input is prepared over the engine's graph batch (MatRIS's is its own graph view):
            # its input check (its element range, in ``prepare``) of the new numbers, before they enter the
            # captured batches in place (a bucket's input was prepared once, at its capture).
            species = self.owner.batch.species.clone()
            species[: self.n_real] = device_numbers
            self.adapter.prepare(dataclasses.replace(self.owner.batch, species=species))
        self._numbers = numbers.copy()
        self._builder_args["atomic_numbers"].copy_(device_numbers)
        for builder in [self.owner, *(entry["builder"] for entry in self.cache.values())]:
            builder.set_atomic_numbers(device_numbers)
        if self._scratch is not None:
            self._scratch.set_atomic_numbers(device_numbers)
        self._event = True

    def _margin(self, u, t):
        """The demand a bucket is selected for: the last known demand (the
        previous call's, or this call's count at a refresh or redo) plus the
        minimum padding, so the usual step-to-step increase does not overflow."""
        return u + self.min_pad_u, t + self._inference_config.capacity.min_pad_t

    def _displacement(self, positions: np.ndarray) -> float:
        """Host minimum-image max displacement since the last refresh (A)."""
        delta = positions - self._reference
        delta -= np.rint(delta @ self._inv_cell) @ self.cell
        return float(np.sqrt(np.einsum("ij,ij->i", delta, delta).max(initial=0.0)))

    def _refresh(self, positions: np.ndarray) -> None:
        """Refresh the candidates at ``positions`` and count their exact demand
        (one small readback; refreshes are rare and start a call)."""
        self.owner.refresh_candidates(self.positions)
        _, status = self.owner.build(self.positions, rebuild_candidates=False)
        capacity_valid, edges, triplets = torch.cat([
            status.candidate_capacity_valid.long(), status.edge_count, status.triplet_count]).tolist()
        if not capacity_valid:
            raise SkinTopologyUnavailable(
                f"a candidate row exceeds {self.owner.buffer_max} slots within {self.owner.candidate_cutoff} A")
        self._reference = np.array(positions, dtype=np.float64, copy=True)
        self._demand = ((edges + 1) // 2, triplets)
        self.refreshes += 1
        self._event = True

    def _launch(self, entry):
        if entry["graph"] is not None:
            with nvtx_range("cudaGraphLaunch"):
                entry["graph"].replay()
            return entry["out"], entry["flags"]
        # Eager: the ambient fusion scope, like the eager rebuild path (the
        # fusion scope enables graph-only kernels).
        return self._body(entry["builder"])

    @torch.no_grad()
    def _read(self, out, flags) -> tuple[list[int], dict]:
        """One readback: status flags and every requested output (fresh host copies)."""
        n = self.n_real
        energy = out["e"].reshape(-1)[:1]
        # A model without reference energies may report none (0).
        ref = out.get("ref_energy", 0.0)
        ref = ref.reshape(-1)[:1] if torch.is_tensor(ref) else torch.full_like(energy, float(ref))
        scalars = [energy, ref]
        if "s" in out:
            scalars.append(out["s"][0].reshape(-1))
        # An FP64 energy (the engine-owned derivative's) is read in FP64, the stress values unchanged.
        wide = energy.dtype == torch.float64
        if wide and self._pinned_scalars64 is None:
            self._pinned_scalars64 = torch.empty(11, dtype=torch.float64, pin_memory=True)
        pinned = self._pinned_scalars64 if wide else self._pinned_scalars
        scalars = torch.cat(scalars).to(pinned.dtype)
        k = scalars.numel()
        self._pinned_flags.copy_(flags, non_blocking=True)
        pinned[:k].copy_(scalars, non_blocking=True)
        if "f" in out:
            self._pinned_forces.copy_(out["f"][0][:n], non_blocking=True)
        if "m" in out:
            self._pinned_magmoms.copy_(out["m"][0][:n].reshape(n), non_blocking=True)
        torch.cuda.current_stream().synchronize()
        values = pinned[:k].numpy()
        if self.capture:
            result = {"e": float(values[0]), "ref_energy": float(values[1])}
        else:
            # Like the eager rebuild path: scalars of the energy's dtype
            # (float32 for MatRIS: calculate() then scales them in float32).
            result = {"e": values[0:1].reshape(()).copy(), "ref_energy": values[1:2].reshape(()).copy()}
        if "s" in out:
            result["s"] = values[2:11].reshape(3, 3).astype(np.float32)
        if "f" in out:
            result["f"] = self._pinned_forces.numpy().copy()
        if "m" in out:
            result["m"] = self._pinned_magmoms.numpy().copy()
        return self._pinned_flags.tolist(), result

    @scoped_call
    def evaluate(self, positions: np.ndarray, atoms=None) -> dict:
        """Energy/forces/(stress, magmoms) of ``positions`` (host, float64).

        ``atoms`` (the ASE Atoms of these positions) is used only by the
        checker. Raises SkinTopologyUnavailable when a candidate row is
        truncated (the caller then rebuilds per call).
        """
        with nvtx_range("SkinTopologyRunner.evaluate"), torch.cuda.device(self.device):
            positions = np.asarray(positions, dtype=np.float64)
            self._pinned_positions.numpy()[...] = positions
            self.positions.copy_(self._pinned_positions, non_blocking=True)
            if self._reference is None:
                self._refresh(positions)
            else:
                displacement = self._displacement(positions)
                self.max_host_displacement = max(self.max_host_displacement, displacement)
                if displacement >= PROACTIVE_REFRESH_FRACTION * self.skin:
                    self._refresh(positions)
            for _ in range(_MAX_ATTEMPTS):
                # A redo must not pin the previous bucket across a pool release.
                entry = result = None
                entry = self._bucket_entry(self.n_real, *self._margin(*self._demand), None)
                flags, result = self._read(*self._launch(entry))
                candidates_valid, capacity_valid, exceeded, edges, triplets, isolated = flags
                if not capacity_valid:
                    raise SkinTopologyUnavailable(
                        f"a candidate row exceeds {self.owner.buffer_max} slots")
                if not candidates_valid:
                    # An atom moved half the skin without the host test seeing
                    # it: the counts are of a stale list, refresh them first.
                    self.redo_skin += 1
                    self._refresh(positions)
                    continue
                if exceeded:
                    # The builder evaluated the all-padding topology: redo in a
                    # bucket that fits this call's exact demand.
                    self._demand = ((edges + 1) // 2, triplets)
                    u_cap, t_cap = entry["builder"].u_capacity, entry["builder"].t_capacity
                    if u_cap > self._demand[0] and t_cap >= self._demand[1]:
                        raise SkinTopologyUnavailable(f"invalid pair topology at demand {self._demand}")
                    self.redo_capacity += 1
                    self._event = True
                    continue
                break
            else:
                raise RuntimeError(f"skin topology: no valid evaluation in {_MAX_ATTEMPTS} attempts")
            self._demand = ((edges + 1) // 2, triplets)
            self.last_edge_count, self.last_triplet_count = edges, triplets
            self.calls += 1
            if self.check_isolated_atoms and isolated:
                raise_if_isolated_atoms(isolated)
            event, self._event = self._event, False
            if self.check_every and (event or self.calls % self.check_every == 0):
                self.check_topology(entry["builder"], positions, atoms)
            return result

    # --- strict checker ----------------------------------------------------

    @torch.no_grad()
    def check_topology(self, builder, positions: np.ndarray, atoms=None) -> None:
        """Compare ``builder``'s last topology with independent builds (debug; syncs).

        (a) A builder with candidates refreshed at ``positions`` (host input,
            not the runner's device copy): bitwise.
        (b) The per-call ragged build of ``atoms`` (topology_reuse="off";
            edges only: ``atoms_to_edges_gpu``): atomic numbers, lattice and
            fractional coordinates (edges only: and the wrapped positions)
            bitwise; the same edges and triplets up to CUTOFF_TIE pairs, and
            bitwise index tensors when the edge and short-edge sets agree.
        """
        self.checks += 1
        n = self.n_real
        edges = int(builder.edge_offsets[-1])
        triplets = int(builder.pair_offsets[-1])

        def view(b, e, t):
            fields = {
                "atomic_number": b.batch.species[:n], "lattice": b.batch.cell,
                "frac": b.frac_coords[:n], "center": b.center[:e], "neighbor": b.neighbor[:e],
                "images": b.images[:e],
            }
            if self._edges_only:
                # The wrapped coordinates the model reads; no pairs or triplets.
                fields["positions"] = b.batch.positions[:n]
            else:
                fields.update(d2u=b.directed2undirected[:e], u2d=b.undirected2directed[: e // 2],
                              line_graph=b.line_graph[:t])
            fields["isolated"] = b._isolated_atom_count
            return fields

        mine = view(builder, edges, triplets)
        host_positions = torch.from_numpy(np.asarray(positions, dtype=np.float32)).to(self.device)
        if self._scratch is None or (self._scratch.u_capacity, self._scratch.t_capacity) != (
                builder.u_capacity, builder.t_capacity):
            self._scratch = FixedCapacityGraphBuilder(
                u_capacity=builder.u_capacity, t_capacity=builder.t_capacity, **self._builder_args)
        fresh = self._scratch
        fresh.build(host_positions, rebuild_candidates=True)
        fresh_counts = (int(fresh.edge_offsets[-1]), int(fresh.pair_offsets[-1]))
        theirs = view(fresh, *fresh_counts)
        bad = [name for name in mine if not torch.equal(mine[name], theirs[name])]
        if (edges, triplets) != fresh_counts or bad:
            raise AssertionError(
                f"skin topology differs from a fresh candidate build at call {self.calls}: "
                f"(edges, triplets) {(edges, triplets)} vs {fresh_counts}, fields {bad}")
        if atoms is None:
            return
        if self._edges_only:
            self._check_edges_only_rebuild(mine, edges, atoms)
            return

        ragged = atoms_to_graph_gpu(atoms, atom_graph_cutoff=self.atom_graph_cutoff,
                                    line_graph_cutoff=self.line_graph_cutoff, check_isolated_atoms=False)
        bad = [name for name, value in (("atomic_number", ragged.atomic_number), ("lattice", ragged.lattice),
                                        ("frac", ragged.atom_frac_coord)) if not torch.equal(value, mine[name])]
        if bad:
            raise AssertionError(f"skin graph inputs differ from the ASE atoms at call {self.calls}: {bad}")
        wrapped = small_matmul(mine["frac"], mine["lattice"])
        same = (
            ragged.atom_graph.shape[0] == edges
            and torch.equal(ragged.atom_graph[:, 0], mine["center"])
            and torch.equal(ragged.atom_graph[:, 1], mine["neighbor"])
            and torch.equal(ragged.neighbor_image, mine["images"].float())
        )
        if not same:
            # Different edge sets: every difference must be a cutoff tie.
            self.check_ties += self._tie_pairs(
                wrapped, torch.stack([mine["center"], mine["neighbor"]], 1), mine["images"],
                ragged.atom_graph, ragged.neighbor_image.round().int(), self.atom_graph_cutoff)
            return
        bad = [name for name, value in (("d2u", ragged.directed2undirected), ("u2d", ragged.undirected2directed))
               if not torch.equal(value, mine[name])]
        if bad:
            raise AssertionError(f"skin topology differs from the per-call build at call "
                                 f"{self.calls} with equal edges: fields {bad}")
        if torch.equal(ragged.line_graph, mine["line_graph"]):
            return
        # Equal edges, different triplets: the short edges (d < r3) may differ
        # only by r3 ties; with equal short edges the triplets must agree. The
        # ragged builder's decision, recomputed with its expression.
        diff = (wrapped[ragged.atom_graph[:, 1].long()] - wrapped[ragged.atom_graph[:, 0].long()]
                + small_matmul(ragged.neighbor_image.unsqueeze(-2), ragged.lattice).squeeze(-2))
        short_theirs = diff.norm(dim=1) < self.line_graph_cutoff
        short_mine = builder.short_flags[:edges]
        if torch.equal(short_mine, short_theirs):
            raise AssertionError(f"skin topology: triplets differ from the per-call build at call "
                                 f"{self.calls} with equal edges and short edges")
        rows = torch.nonzero(short_mine != short_theirs).flatten()
        pairs = torch.stack([mine["center"], mine["neighbor"]], 1)[rows]
        self.check_ties += self._tie_pairs(wrapped, pairs, mine["images"][rows],
                                           pairs[:0], mine["images"][:0], self.line_graph_cutoff)

    def _check_edges_only_rebuild(self, mine, edges, atoms) -> None:
        """check_topology (b) of an edges-only model against ``atoms_to_edges_gpu`` of ``atoms`` (its
        per-call build): the same inputs and wrapped positions bitwise, the same edges up to
        CUTOFF_TIE pairs."""
        ragged = atoms_to_edges_gpu(atoms, self.atom_graph_cutoff, self.device, check_isolated_atoms=False)
        bad = [name for name, value in (("atomic_number", ragged.species), ("lattice", ragged.cell),
                                        ("frac", ragged.frac_coords), ("positions", ragged.positions))
               if not torch.equal(value, mine[name])]
        if bad:
            raise AssertionError(f"skin graph inputs differ from the ASE atoms at call {self.calls}: {bad}")
        if (ragged.edge_index.shape[0] == edges
                and torch.equal(ragged.edge_index[:, 0], mine["center"])
                and torch.equal(ragged.edge_index[:, 1], mine["neighbor"])
                and torch.equal(ragged.edge_image, mine["images"].float())):
            return
        # Different edge sets: every difference must be a cutoff tie.
        self.check_ties += self._tie_pairs(
            small_matmul(mine["frac"], mine["lattice"]), torch.stack([mine["center"], mine["neighbor"]], 1),
            mine["images"], ragged.edge_index, ragged.edge_image.round().int(), self.atom_graph_cutoff)

    def _tie_pairs(self, wrapped, edges_a, images_a, edges_b, images_b, cutoff) -> int:
        """Pairs in exactly one edge set; raises unless each is a cutoff tie."""
        def keys(edges, images):
            return {tuple(row) for row in torch.cat([edges.long(), images.long()], 1).tolist()}

        diff = keys(edges_a, images_a) ^ keys(edges_b, images_b)
        if not diff:
            return 0
        rows = torch.tensor(sorted(diff), dtype=torch.long, device=wrapped.device)
        vector = (wrapped[rows[:, 1]].double() - wrapped[rows[:, 0]].double()
                  + small_matmul(rows[:, 2:].double(), self._builder_args["cell"].double()))
        gap = (torch.linalg.vector_norm(vector, dim=1) - cutoff).abs().max().item()
        if gap > CUTOFF_TIE:
            raise AssertionError(f"skin topology: {len(diff)} pairs differ from the per-call build "
                                 f"at call {self.calls}, up to {gap:.3g} A from the {cutoff} A cutoff")
        return len(diff)

    def stats(self) -> dict[str, object]:
        stats = super().stats()
        stats.update(
            capture=self.capture, skin=self.skin, calls=self.calls, refreshes=self.refreshes,
            redo_capacity=self.redo_capacity, redo_skin=self.redo_skin, checks=self.checks,
            check_ties=self.check_ties, max_host_displacement=self.max_host_displacement,
            last_edge_count=self.last_edge_count, last_triplet_count=self.last_triplet_count,
            buffer_max=self.owner.buffer_max,
        )
        return stats
