"""Topology contract IR: builder evidence, verifier rejections and lowering selection.

The lowering-selection tests (which lowering a certified MatRIS or M3GNet
topology selects) are structural perf guards: every lowering is numerically
correct, so they run only in the opt-in perf tier (``MATRIS_RUN_PERF_TIER=1``,
tests/matris_test_tiers.py). The M3GNet tests need MatGL.
"""
from __future__ import annotations

from types import SimpleNamespace
import tempfile
import unittest

import torch

from fastmd.graph.matris_topology import (
    EDGE_SOURCE,
    EDGE_TARGET,
    TRIPLET_OWNER,
    matris_builder_evidence,
)
from fastmd.graph.radiusgraph import RadiusGraph
from fastmd.graph.topology_contract import (
    ContractProperty,
    DomainExtent,
    DomainSpec,
    PropertyKind,
    RelationSpec,
    TopologyDomain,
    TopologyInstance,
    TopologySchema,
    TopologyVerificationError,
    VerificationMode,
    certify_topology,
)
from fastmd.models.m3gnet import bind_m3gnet_topology
from fastmd.engine.topology_lowering import compile_lowering_plan
from fastmd.models.matris.topology import bind_matris_topology, matris_lowering_requests
from matris_test_tiers import perf_tier

try:
    import matgl  # noqa: F401

    HAS_MATGL = True
except ImportError:
    HAS_MATGL = False


def _matris_binding():
    edges = torch.tensor([[0, 1], [0, 2], [1, 0], [2, 0]], dtype=torch.long)
    d2p = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    p2d = torch.tensor([0, 1], dtype=torch.long)
    line = torch.tensor(
        [[0, 0, 0, 1, 1], [0, 1, 1, 0, 0]], dtype=torch.long
    )
    evidence = matris_builder_evidence(
        "test.matris",
        atom_target_sorted=True,
        line_owner_sorted=True,
    )
    graph = RadiusGraph(
        graph_id=None,
        mp_id=None,
        composition="H3",
        atomic_number=torch.ones(3, dtype=torch.int32),
        atom_frac_coord=torch.zeros((3, 3)),
        lattice=torch.eye(3),
        neighbor_image=torch.zeros((4, 3)),
        atom_graph=edges,
        atom_graph_cutoff=5.0,
        line_graph=line,
        line_graph_cutoff=4.0,
        directed2undirected=d2p,
        undirected2directed=p2d,
        atom_target_sorted=True,
        line_atom_sorted=True,
        topology_evidence=evidence,
    )
    batched = {
        "atomic_numbers": graph.atomic_number,
        "atom_segment": torch.zeros(3, dtype=torch.long),
        "num_graphs": 1,
        "directed2undirected": d2p,
        "undirected2directed": p2d,
        "atom_graph_dict": {
            "target_index": edges[:, 0],
            "source_index": edges[:, 1],
            "num_segment": 3,
            "num_undirected": 2,
        },
        "line_graph_dict": {
            "line_graph": line[:, [0, 1, 3]],
            "atom_list": line[:, 0],
            "target_index": line[:, 1],
            "source_index": line[:, 3],
            "target_DE_index": line[:, 2],
            "source_DE_index": line[:, 4],
        },
    }
    return bind_matris_topology(
        batched, [graph], mode=VerificationMode.EXHAUSTIVE
    )


def _corrupt(instance: TopologyInstance, relation: str, value: torch.Tensor):
    relations = dict(instance.relations)
    relations[relation] = value
    return TopologyInstance(
        instance.schema,
        instance.extents,
        relations,
        producer="corruption-test",
    )


class TopologyContractTest(unittest.TestCase):
    def test_builder_evidence_uses_weights_only_safe_serialization(self) -> None:
        evidence = matris_builder_evidence(
            "test.serialization",
            atom_target_sorted=True,
            line_owner_sorted=True,
        )
        graph = RadiusGraph(
            None,
            None,
            "H2",
            torch.ones(2, dtype=torch.int32),
            torch.zeros((2, 3)),
            torch.eye(3),
            torch.zeros((2, 3)),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
            2.0,
            torch.empty((0, 5), dtype=torch.int32),
            1.5,
            torch.tensor([0, 0], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            topology_evidence=evidence,
        )
        with tempfile.NamedTemporaryFile(suffix=".pt") as file:
            graph.save(file.name)
            payload = torch.load(file.name)
        restored = RadiusGraph.from_dict(payload)
        self.assertEqual(restored.topology_evidence, evidence)

    @perf_tier
    def test_matris_contract_selects_all_migrated_lowerings(self) -> None:
        binding = _matris_binding()
        self.assertTrue(binding.report.passed)
        instance = binding.instance
        plan = compile_lowering_plan(
            instance.schema,
            instance.certificate,
            matris_lowering_requests(),
        )
        selected = {item.request_key: item.lowering for item in plan.selections}
        self.assertEqual(
            selected,
            {
                "atom_indexed_affine": "indexed_affine.decomposed_sorted_vjp",
                "line_indexed_affine": "indexed_affine.decomposed_sorted_vjp",
                "target_segment_attention": "segment_attention.identity_csr",
                "line_target_segment_attention": "segment_attention.sorted_csr",
                "pair_algebra": "pair_algebra.paired_rows",
                "line_incidence": "line_incidence.direct",
                "line_envelope": "line_incidence.direct",
                "sorted_segment_vjp": "segment_vjp.contiguous",
            },
        )

    def test_verifier_rejects_each_corruption_by_its_failed_check(self) -> None:
        instance = _matris_binding().instance
        # A minimal real/dummy instance: valid as built, corrupted below.
        closure_schema = TopologySchema(
            domains=(
                DomainSpec("edge", TopologyDomain.DIRECTED_EDGE, has_real_prefix=True),
                DomainSpec("atom", TopologyDomain.ATOM, has_real_prefix=True),
            ),
            relations=(RelationSpec("target", "edge", "atom"),),
            properties=(
                ContractProperty(PropertyKind.BOUNDED, "target"),
                ContractProperty(PropertyKind.REAL_DUMMY_CLOSED, "target"),
            ),
        )
        closure = TopologyInstance(
            closure_schema,
            {
                "edge": DomainExtent(2, 2, 3),
                "atom": DomainExtent(2, 2, 3),
            },
            {"target": torch.tensor([0, 1, 2])},
            "test",
        )
        certified, _ = certify_topology(closure, mode=VerificationMode.EXHAUSTIVE)
        self.assertIsNotNone(certified.certificate)
        exhaustive = VerificationMode.EXHAUSTIVE
        # (case, instance, mode, the structured check that must fail)
        cases = (
            (
                EDGE_TARGET,
                _corrupt(instance, EDGE_TARGET, torch.tensor([1, 0, 1, 2])),
                exhaustive,
                "SortedBy(edge_target)",
            ),
            (
                "directed_to_pair",
                _corrupt(instance, "directed_to_pair", torch.tensor([0, 0, 0, 1])),
                exhaustive,
                "PairCardinality(directed_to_pair)",
            ),
            (
                EDGE_SOURCE,
                _corrupt(instance, EDGE_SOURCE, torch.tensor([1, 2, 2, 0])),
                exhaustive,
                "ReversePairInvolution(directed_to_pair)",
            ),
            (
                TRIPLET_OWNER,
                _corrupt(instance, TRIPLET_OWNER, torch.tensor([1, 1])),
                exhaustive,
                "Incidence(triplet_owner)",
            ),
            (
                "triplet_edge_a",
                _corrupt(instance, "triplet_edge_a", torch.tensor([99, 1])),
                exhaustive,
                "Incidence(triplet_owner)",
            ),
            (
                EDGE_SOURCE,
                _corrupt(instance, EDGE_SOURCE, torch.tensor([1, 3, 0, 0])),
                exhaustive,
                "Bounded(edge_source)",
            ),
            # BUILDER mode trusts only producer evidence, which this lacks.
            (
                "producer",
                TopologyInstance(
                    instance.schema,
                    instance.extents,
                    instance.relations,
                    producer="unproven",
                ),
                VerificationMode.BUILDER,
                "builder evidence supplied",
            ),
            # A real edge targeting a dummy atom.
            (
                "target",
                _corrupt(closure, "target", torch.tensor([0, 1, 0])),
                exhaustive,
                "RealDummyClosed(target)",
            ),
        )
        for case, corrupted, mode, expected_check in cases:
            with self.subTest(case=case, check=expected_check):
                with self.assertRaises(TopologyVerificationError) as caught:
                    certify_topology(corrupted, mode=mode)
                failed = {
                    check.name
                    for check in caught.exception.report.checks
                    if not check.passed
                }
                self.assertIn(expected_check, failed)

    @perf_tier
    @unittest.skipUnless(HAS_MATGL, "MatGL is an optional validation dependency")
    def test_m3gnet_reuses_lowering_and_applies_joint_cost_guard(self) -> None:
        edge_index = torch.tensor(
            [[0, 0, 1, 1], [1, 2, 0, 2]], dtype=torch.long
        )
        graph = SimpleNamespace(
            edge_index=edge_index,
            pos=torch.zeros((3, 3)),
            batch=torch.zeros(3, dtype=torch.long),
            num_graphs=1,
        )
        line_graph = {
            "line_edge_index": torch.tensor(
                [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long
            ),
            "n_triple_ij": torch.ones(4, dtype=torch.long),
            "kept_edge_ids": torch.arange(4),
        }
        m3gnet_64 = bind_m3gnet_topology(
            graph, line_graph, mode=VerificationMode.EXHAUSTIVE
        )
        for key in ("edge_indexed_affine", "node_indexed_affine"):
            self.assertEqual(
                m3gnet_64.plan.selection(key).lowering,
                "indexed_affine.materialized",
            )

        m3gnet_128 = bind_m3gnet_topology(
            graph,
            line_graph,
            mode=VerificationMode.EXHAUSTIVE,
            feature_dim=128,
        )
        matris = _matris_binding().instance
        matris_plan = compile_lowering_plan(
            fastmd.schema,
            fastmd.certificate,
            matris_lowering_requests(),
        )
        matris_rule = matris_plan.selection("atom_indexed_affine")
        for key in ("edge_indexed_affine", "node_indexed_affine"):
            m3gnet_rule = m3gnet_128.plan.selection(key)
            self.assertEqual(m3gnet_rule.lowering, matris_rule.lowering)
            self.assertEqual(m3gnet_rule.forward_rule, matris_rule.forward_rule)
            self.assertEqual(m3gnet_rule.vjp_rule, matris_rule.vjp_rule)
        self.assertEqual(
            m3gnet_64.plan.selection("threebody_incidence").lowering,
            "line_incidence.direct",
        )

    @unittest.skipUnless(HAS_MATGL, "MatGL is an optional validation dependency")
    def test_m3gnet_cached_segment_preserves_forward_and_vjp(self) -> None:
        from matgl.layers import ThreeBodyInteractions
        from matgl.utils.maths import get_segment_indices_from_n

        from fastmd.models.m3gnet import (
            _ContractThreeBodyInteractions,
            _matgl_segment_ids_on_device,
        )

        torch.manual_seed(3)
        atom_dim, basis_dim, edge_dim = 5, 3, 4
        base = ThreeBodyInteractions(
            update_network_atom=torch.nn.Linear(atom_dim, basis_dim),
            update_network_bond=torch.nn.Linear(basis_dim, edge_dim),
        )
        counts = torch.tensor([2, 0, 1], dtype=torch.long)
        segment = get_segment_indices_from_n(counts)
        torch.testing.assert_close(
            _matgl_segment_ids_on_device(counts, int(counts.sum())), segment
        )
        wrapped = _ContractThreeBodyInteractions(base, segment)
        edge_dst_atom = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        line_edge_index = torch.tensor(
            [[0, 0, 2], [1, 2, 0]], dtype=torch.long
        )

        def inputs():
            return tuple(
                value.requires_grad_()
                for value in (
                    torch.randn(3, basis_dim),
                    torch.rand(4),
                    torch.randn(3, atom_dim),
                    torch.randn(4, edge_dim),
                )
            )

        basis_a, cutoff_a, node_a, edge_a = inputs()
        baseline = base(
            edge_dst_atom,
            line_edge_index,
            counts,
            4,
            basis_a,
            cutoff_a,
            node_a,
            edge_a,
        )
        cotangent = torch.randn_like(baseline)
        baseline_grads = torch.autograd.grad(
            baseline, (basis_a, cutoff_a, node_a, edge_a), cotangent
        )

        basis_b = basis_a.detach().clone().requires_grad_()
        cutoff_b = cutoff_a.detach().clone().requires_grad_()
        node_b = node_a.detach().clone().requires_grad_()
        edge_b = edge_a.detach().clone().requires_grad_()
        lowered = wrapped(
            edge_dst_atom,
            line_edge_index,
            counts,
            4,
            basis_b,
            cutoff_b,
            node_b,
            edge_b,
        )
        lowered_grads = torch.autograd.grad(
            lowered, (basis_b, cutoff_b, node_b, edge_b), cotangent
        )
        torch.testing.assert_close(lowered, baseline, rtol=1e-6, atol=1e-6)
        for actual, expected in zip(lowered_grads, baseline_grads, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

        # MatGL's scatter helper returns zero rows when there are no triplets,
        # so the interaction returns the original bond features directly.
        empty_counts = torch.tensor([0], dtype=torch.long)
        empty_line_edges = torch.empty((2, 0), dtype=torch.long)
        empty_basis = torch.empty((0, basis_dim))
        one_cutoff = torch.ones(1)
        one_node = torch.randn(1, atom_dim)
        one_edge = torch.randn(1, edge_dim)
        empty_segment = _matgl_segment_ids_on_device(empty_counts, 0)
        empty_wrapped = _ContractThreeBodyInteractions(base, empty_segment)
        expected = base(
            torch.tensor([0]),
            empty_line_edges,
            empty_counts,
            1,
            empty_basis,
            one_cutoff,
            one_node,
            one_edge,
        )
        actual = empty_wrapped(
            torch.tensor([0]),
            empty_line_edges,
            empty_counts,
            1,
            empty_basis,
            one_cutoff,
            one_node,
            one_edge,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
