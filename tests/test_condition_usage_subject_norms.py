from unittest.mock import patch

import torch

from analysis import condition_usage


def test_norms_are_assigned_per_subject_in_mixed_batch():
    class Model:
        def eval(self):
            return self

        def _prepare_condition(self, _unused, condition, _other):
            return condition['cond_z']

        def __call__(self, x, sc, *args, **kwargs):
            return None, {'dynamics_pred': torch.zeros(x.shape[0], x.shape[1], 1)}

    view = []
    for subject, value in [('a', 1.0), ('b', 3.0)]:
        view.append({
            'subj_id': subject, 'x': torch.zeros(2, 1, 2), 'sc': torch.zeros(2, 2),
            'y': torch.ones(2, 1), 'x_last': torch.zeros(2),
            'cond_z': torch.tensor([value]), 'cond_code': torch.zeros(1, dtype=torch.long),
            'cond_mask': torch.ones(1),
        })

    def film(_model, embedding):
        return {'ode_film_gamma': embedding.mean().item(),
                'ode_film_beta': embedding.mean().item()}

    def adarms(_model, embedding):
        return {'encoder_adarms_gamma': embedding.mean().item(),
                'encoder_adarms_beta': embedding.mean().item()}

    with patch('main.dynamics_extras', return_value=(None, None)), \
            patch.object(condition_usage, 'film_norms', side_effect=film), \
            patch.object(condition_usage, 'adarms_norms', side_effect=adarms):
        rows = condition_usage.run_condition_usage(
            Model(), None, ['a', 'b'], view, torch.device('cpu'), 2, 0)

    assert [row['ode_film_gamma'] for row in rows] == [1.0, 3.0]
    assert [row['encoder_adarms_gamma'] for row in rows] == [1.0, 3.0]
