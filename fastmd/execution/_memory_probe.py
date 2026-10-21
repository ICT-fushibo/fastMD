"""Memory runs of a model other than MatRIS, for ``fastmd.adapters.memory.MeasuredMemoryModel``.

K6 (``fastmd.engine.memory_admission``) has no calibration for such a model, so execution auto
measures it instead: the model's own evaluation on a fixed-capacity build of the system (the runtime's
candidate list: cutoff + skin; edges only, or with the line graph of a model with triplets), once eager
and once captured in a CUDA graph, at a few capacities below the one its capture would hold. Each run
makes its own builder, and a captured graph is released before the run returns, so that every point
starts from the same state.
"""
from __future__ import annotations

import torch

from fastmd.engine._graph_capture import capture_graph

# The measured capacities, as fractions of the capacity the capture would hold: the prediction there stays
# within twice the largest measured one, and the probe itself never allocates the full capture.
PROBE_FRACTIONS = (0.25, 0.5, 0.75)


# With triplets: (edge, triplet) fractions of the targets. Their triplet/edge ratios are half, one and two
# times the targets' (the fit tells the per-edge and per-triplet costs apart, and the target's ratio lies
# within the measured ones), and the largest stays at 0.75 of either target, as for edges only.
PROBE_PAIR_FRACTIONS = ((0.25, 0.5), (0.5, 0.25), (0.5, 0.5), (0.75, 0.75))


def probe_capacities(target_edges: int) -> list[int]:
    """Edge capacities (whole undirected pairs, at least one) at ``PROBE_FRACTIONS`` of ``target_edges``."""
    return sorted({2 * max(1, int(fraction * target_edges) // 2) for fraction in PROBE_FRACTIONS})


def probe_capacity_pairs(target_edges: int, target_triplets: int) -> list[tuple[int, int]]:
    """(edge, triplet) capacities at ``PROBE_PAIR_FRACTIONS`` of the targets (whole undirected pairs, at
    least one of each)."""
    return sorted({(2 * max(1, int(edges * target_edges) // 2), max(1, int(triplets * target_triplets)))
                   for edges, triplets in PROBE_PAIR_FRACTIONS})


def edges_only_memory_runs(adapter, *, cell, atomic_numbers, pbc, positions, n_real: int, device, task: str = "ef",
                           candidate_skin: float = 0.0, n_dummy: int = 64):
    """``{kind: run(edge capacity)}`` for ``MeasuredMemoryModel.measure``: ``adapter``'s evaluation of
    ``positions`` (``task``) on an edges-only fixed-capacity build at that capacity (the adapter's padding
    rule and ``prepare``, and the builder options the runtime gives the model; candidates within cutoff +
    ``candidate_skin``, built outside any graph), eager ("eager") or captured in a CUDA graph after a
    warmup on the capture stream ("capture"). The evaluation is the model's own call, or the
    engine-owned derivative of a model with an energy tap. Each run returns the (edges, triplets)
    capacity it held. Only for a model declared sync-free (a capture of its evaluation must be valid)."""
    from fastmd.engine.energy_derivative import edges_only_builder_options

    def capacities(capacity: int) -> tuple[int, int]:
        return max(1, capacity // 2), 0

    return _memory_runs(adapter, capacities, edges_only_builder_options(adapter), cell=cell,
                        atomic_numbers=atomic_numbers, pbc=pbc, positions=positions, n_real=n_real, device=device,
                        task=task, candidate_skin=candidate_skin, n_dummy=n_dummy)


def line_graph_memory_runs(adapter, *, cell, atomic_numbers, pbc, positions, n_real: int, device, task: str = "ef",
                           candidate_skin: float = 0.0, n_dummy: int = 64):
    """:func:`edges_only_memory_runs` for a model with a line graph that the engine differentiates: its
    evaluation on a fixed-capacity build with the line graph (the adapter's padding rule, the positions published), at an
    (edge, triplet) capacity pair; each run returns the (edges, triplets) capacity it held."""
    from fastmd.engine.energy_derivative import engine_derivative

    if engine_derivative(adapter) is None:
        raise ValueError("line_graph_memory_runs measures a model the engine differentiates")

    def capacities(capacity: tuple[int, int]) -> tuple[int, int]:
        return max(1, capacity[0] // 2), max(1, capacity[1])

    return _memory_runs(adapter, capacities, {"publish_positions": True}, cell=cell, atomic_numbers=atomic_numbers,
                        pbc=pbc, positions=positions, n_real=n_real, device=device, task=task,
                        candidate_skin=candidate_skin, n_dummy=n_dummy)


def _memory_runs(adapter, capacities, options: dict, *, cell, atomic_numbers, pbc, positions, n_real: int, device,
                 task: str, candidate_skin: float, n_dummy: int):
    """The runs of the probes above: a builder per run at ``capacities(capacity)`` (undirected pairs,
    triplets), with the builder ``options``."""
    from fastmd.graph.fixed_capacity_builder import FixedCapacityGraphBuilder
    from fastmd.engine.energy_derivative import engine_derivative

    topology = adapter.topology_spec()
    cutoff = float(topology.cutoff)
    line_graph_cutoff = None if topology.triplet_cutoff is None else float(topology.triplet_cutoff)
    derivative = engine_derivative(adapter)
    stream = torch.cuda.Stream(device)

    def builder(capacity):
        u_capacity, t_capacity = capacities(capacity)
        made = FixedCapacityGraphBuilder(
            cell=cell, atomic_numbers=atomic_numbers, pbc=pbc, u_capacity=u_capacity, t_capacity=t_capacity,
            atom_graph_cutoff=cutoff, line_graph_cutoff=line_graph_cutoff,
            candidate_cutoff=cutoff + candidate_skin if candidate_skin > 0 else None, device=device,
            n_dummy=n_dummy, padding=adapter.padding_rule(), prepare=adapter.prepare, **options)
        made.build(positions)  # the candidate list, outside any graph
        return made

    def evaluate(made) -> None:
        inputs, _ = made.build(positions, rebuild_candidates=False)
        if derivative is None:
            adapter.evaluate([inputs], task=task, is_training=False, n_real=n_real)
        else:
            derivative(made.batch, inputs, task=task, n_real=n_real)

    def eager(capacity) -> tuple[int, int]:
        made = builder(capacity)
        evaluate(made)
        return made.e_capacity, made.t_capacity

    def capture(capacity) -> tuple[int, int]:
        made = builder(capacity)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            evaluate(made)
        torch.cuda.current_stream(device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with capture_graph(graph, stream=stream):
            evaluate(made)
        del graph  # every point starts from the same state
        return made.e_capacity, made.t_capacity

    return {"eager": eager, "capture": capture}


def measured_memory(model_class, runs, target_edges: int, device, target_triplets: int | None = None):
    """``model_class.measure`` (``MeasuredMemoryModel``) of ``runs`` at ``probe_capacities(target_edges)``
    (with ``target_triplets``: at ``probe_capacity_pairs``, fitting a per-triplet cost), or
    ``model_class.failed`` with the reason where the measurement ran out of CUDA memory or a capture
    failed (execution auto then stays eager, with that reason); other errors propagate."""
    triplets = target_triplets is not None
    capacities = (probe_capacity_pairs(target_edges, target_triplets) if triplets
                  else probe_capacities(target_edges))
    try:
        return model_class.measure(runs, capacities, device, triplets=triplets)
    except torch.OutOfMemoryError as error:
        why = f"ran out of CUDA memory ({str(error).splitlines()[0][:160]})"
    except RuntimeError as error:
        if "capture" not in str(error).lower():
            raise
        why = f"failed in a capture ({str(error).splitlines()[0][:160]})"
    # Outside the except blocks: a traceback would pin the failed run's graph and buffers.
    return model_class.failed(f"the memory measurement {why}", triplets=triplets)
