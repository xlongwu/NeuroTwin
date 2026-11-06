from unittest.mock import patch
import torch
from analysis import condition_usage


def test_norms_are_assigned_per_subject_in_mixed_batch():
    class Model:
        def eval(self):return self
        def _prepare_condition(self,condition,_epoch):return condition['cond_z']
        def __call__(self,x,condition=None):return torch.zeros(x.shape[0],x.shape[1],32),{}
    view=[]
    for subject,value in [('a',1.),('b',3.)]:
        view.append(dict(subj_id=subject,x=torch.zeros(2,1,64),sc=torch.zeros(2,2),
            y=torch.ones(2,1),x_last=torch.zeros(2),cond_z=torch.tensor([value]),
            cond_code=torch.zeros(1,dtype=torch.long),cond_mask=torch.ones(1)))
    def film(_model,embedding):
        return dict(cpm_film_gamma=embedding.mean().item(),cpm_film_beta=embedding.mean().item())
    with patch.object(condition_usage,'film_norms',side_effect=film):
        rows=condition_usage.run_condition_usage(Model(),None,['a','b'],view,'cpu',2,0)
    assert [r['cpm_film_gamma'] for r in rows]==[1.,3.]
