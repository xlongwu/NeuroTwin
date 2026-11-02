"""Cross-window leakage and transfer checks on known stable/unstable signals."""
import numpy as np

from analysis.repeatable_information import nested_transfer, transfer_kernels
from analysis.information_readout import site_auc, site_folds


def cohort():
    rng=np.random.default_rng(2024)
    sites=np.repeat(['A','B','C','D','E','F'],20)
    labels=np.tile(np.repeat([0,1],10),6)
    values=np.column_stack((rng.integers(1,3,120),rng.normal(35,5,120),rng.normal(12,2,120)))
    return rng,dict(sites=sites,labels=labels,values=values,mask=np.ones_like(values),idx={'Sex':0,'Age':1,'education':2})


def test_target_window_cannot_change_source_kernel_or_model_selection():
    rng,meta=cohort();windows=rng.normal(size=(120,2,4))
    train,valid=site_folds(meta['sites'])[0]
    original,_=transfer_kernels(windows,meta,train,valid,0,False)
    windows[:,1]+=1000
    changed,_=transfer_kernels(windows,meta,train,valid,0,False)
    for reader in original:
        for a,b in zip(original[reader],changed[reader]):
            np.testing.assert_array_equal(a,b)


def test_stable_brain_signal_transfers_both_directions_and_sites_are_disjoint():
    rng,meta=cohort();windows=rng.normal(0,.2,(120,2,4))
    windows[:,:,0]+=3*meta['labels'][:,None]
    for source in (0,1):
        pred,selections=nested_transfer(windows,meta,source)
        for row in selections:
            assert set(row['train_sites']).isdisjoint(row['valid_sites'])
        for reader in pred:
            score=pred[reader]['joint'][:,1-source]
            assert site_auc(meta['labels'],score,meta['sites']).mean()>.9
            np.testing.assert_allclose(pred[reader]['baseline'][:,0],pred[reader]['baseline'][:,1])


def test_source_only_signal_fails_target_transfer():
    rng,meta=cohort();windows=rng.normal(0,.2,(120,2,4))
    windows[:,0,0]+=3*meta['labels']
    pred,_=nested_transfer(windows,meta,0)
    for reader in pred:
        source=site_auc(meta['labels'],pred[reader]['joint'][:,0],meta['sites']).mean()
        target=site_auc(meta['labels'],pred[reader]['joint'][:,1],meta['sites']).mean()
        assert source>.9 and target<.75
