"""Core invariants of the shared-model response comparison, using synthetic data."""
import numpy as np
import pytest
import torch

from analysis.shared_mechanism import (NETWORKS, fixed_effective_graph, network_responses,
                                       nonoverlap_anchors, outgoing_response, permutation_indices)
from analysis.shared_mechanism_summary import adjusted_group_test, nuisance_design, partial_correlation


def test_disjoint_windows_require_enough_observations():
    np.testing.assert_array_equal(nonoverlap_anchors(129), [63,127])
    with pytest.raises(ValueError, match='T>=129'):
        nonoverlap_anchors(128)


def test_common_permutation_preserves_blocks_and_last_frame():
    ix = permutation_indices()
    np.testing.assert_array_equal(ix, permutation_indices())
    assert np.all(ix[:,-1] == 63)
    for start in range(0,64,8):
        np.testing.assert_array_equal(np.sort(ix[:,start:start+8], axis=1),
                                      np.broadcast_to(np.arange(start,start+8), (116,8)))


def test_self_exclusion_and_target_weighting():
    labels = np.array([network for i, network in enumerate(NETWORKS) for _ in range(i+2)])
    response = np.broadcast_to(np.arange(len(labels), dtype=float), (len(labels),len(labels))).copy()
    np.fill_diagonal(response, 1e9)
    mask = ~np.eye(len(labels), dtype=bool)
    expected = [response[mask & (labels[:,None] == network)].mean() for network in NETWORKS]
    np.testing.assert_allclose(outgoing_response(response, labels), expected)
    blocks = network_responses(response, labels)
    assert blocks.max() < len(labels)
    # Unequal target-network sizes must not turn the primary endpoint into an equal-block mean.
    assert not np.allclose(blocks.mean(axis=1), expected)


def test_graph_hook_restores_module_even_after_exception():
    class Model:
        sc_prior = torch.nn.Identity()
    model = Model()
    graph = {'A_eff':torch.eye(2)[None]}
    with pytest.raises(RuntimeError):
        with fixed_effective_graph(model, graph):
            assert model.sc_prior(torch.ones(1)) is graph
            raise RuntimeError('intentional test')
    assert torch.equal(model.sc_prior(torch.ones(1)), torch.ones(1))


def synthetic_cohort():
    rng = np.random.default_rng(71)
    n = 160
    sites = np.repeat(['A','B','C','D'],40)
    group = np.tile([0.,1.],80)
    age = rng.normal(40+8*group,7)
    sex = rng.integers(0,2,n)
    return rng, group, sites, nuisance_design(age,sex,sites)


def test_group_effect_matches_direct_ols_and_removes_nuisance():
    rng, group, sites, c = synthetic_cohort()
    nuisance_only = c @ np.arange(c.shape[1], dtype=float)
    y = np.column_stack([nuisance_only, nuisance_only+2*group+rng.normal(0,.1,len(group))])
    result = adjusted_group_test(y, group, c, sites, n_perm=199)
    direct = np.linalg.lstsq(np.column_stack([c,group]), y, rcond=None)[0][-1]
    np.testing.assert_allclose(result['beta'], direct, atol=1e-12)
    assert result['p'][0] == 1
    assert result['p'][1] == .005
    assert result['ci_low'][1] < 2 < result['ci_high'][1]
    repeated = adjusted_group_test(y, group, c, sites, n_perm=199)
    np.testing.assert_array_equal(result['p'], repeated['p'])


def test_partial_correlation_removes_shared_age_site_signal():
    rng, group, sites, c = synthetic_cohort()
    x = c @ np.ones(c.shape[1]) + rng.normal(size=len(group))
    y = 3*x + c @ np.arange(c.shape[1]) + rng.normal(0,.1,len(group))
    r, p, dof = partial_correlation(y[:,None],x,c)
    assert r[0] > .99 and p[0] < 1e-10
    assert dof == len(group)-np.linalg.matrix_rank(c)-1
