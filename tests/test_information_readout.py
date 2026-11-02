"""Leakage boundaries and known-signal checks for frozen-representation readouts."""
import numpy as np

from analysis.information_readout import (adjusted_features, kernel_scores, nested_readout,
                                         site_auc, site_folds)
from analysis.information_nuisance import demographic_features
from analysis.information_increment import nested_increment


def cohort():
    rng=np.random.default_rng(2024)
    sites=np.repeat(['A','B','C','D','E','F'],20)
    labels=np.tile(np.repeat([0,1],10),6)
    n=len(labels)
    values=np.column_stack((rng.integers(1,3,n),rng.normal(35,5,n),rng.normal(12,2,n)))
    return rng,dict(labels=labels,sites=sites,values=values,mask=np.ones_like(values),
                    idx={'Sex':0,'Age':1,'education':2})


def test_outer_and_inner_sites_never_overlap():
    _,meta=cohort()
    for train,valid in site_folds(meta['sites']):
        assert set(meta['sites'][train]).isdisjoint(meta['sites'][valid])
        for it,iv in site_folds(meta['sites'][train]):
            assert set(meta['sites'][train[it]]).isdisjoint(meta['sites'][train[iv]])


def test_held_out_values_do_not_change_training_preprocessing():
    rng,meta=cohort()
    x=rng.normal(size=(120,15))
    train,valid=site_folds(meta['sites'])[0]
    original=adjusted_features(x,meta,train)
    changed_x=x.copy();changed_x[valid]+=1000
    changed_meta={**meta,'values':meta['values'].copy()}
    changed_meta['values'][valid,1]+=1000
    changed=adjusted_features(changed_x,changed_meta,train)
    np.testing.assert_array_equal(original[train],changed[train])
    assert not np.allclose(original[valid],changed[valid])


def test_weighted_kernel_solver_matches_primal_ridge():
    rng=np.random.default_rng(13)
    x=rng.normal(size=(24,5));test=rng.normal(size=(8,5))
    y=np.tile([0,1],12);weights=rng.uniform(.5,2,len(y));alpha=.3
    result=kernel_scores(x@x.T,test@x.T,y,weights,[alpha])[:,0]
    coef=np.linalg.solve(x.T@(weights[:,None]*x)+alpha*np.eye(5),x.T@(weights*(2*y-1)))
    np.testing.assert_allclose(result,test@coef,atol=1e-12)


def test_nested_probes_recover_known_signal_and_preserve_score_direction():
    rng,meta=cohort()
    x=rng.normal(0,.2,(120,4))
    x[:,0]+=3*meta['labels']
    predictions,selections,fold=nested_readout(x,meta)
    assert len(selections)==6 and set(fold)=={0,1,2}
    for pred in predictions.values():
        assert site_auc(meta['labels'],pred,meta['sites']).mean()>.95
        assert site_auc(meta['labels'],-pred,meta['sites']).mean()<.05


def test_demographic_baseline_never_fits_held_out_values():
    _,meta=cohort()
    train,valid=site_folds(meta['sites'])[0]
    original=demographic_features(meta,train)
    modified={**meta,'values':meta['values'].copy()}
    modified['values'][valid,1]+=10000
    changed=demographic_features(modified,train)
    np.testing.assert_array_equal(original[train],changed[train])


def test_joint_probe_can_recover_signal_independent_of_demographics():
    rng,meta=cohort()
    x=rng.normal(0,.2,(120,4))
    x[:,0]+=3*meta['labels']
    predictions,_=nested_increment(x,meta)
    for pred in predictions.values():
        assert site_auc(meta['labels'],pred,meta['sites']).mean()>.9
