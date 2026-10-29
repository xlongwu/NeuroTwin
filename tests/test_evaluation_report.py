# coding=utf-8
"""``analysis/evaluation_report.py`` 的验收测试（合成 eval 产物，CPU，不落仓库目录）。

覆盖：
  1. 正常路径：带 ``obs_*`` 条件列的 eval 目录能出全部 7 张图与 ``summary.md``；
  2. 缺产物时 fail-closed，报错点名缺失文件而不是静默出半套图；
  3. 条件未启用（无 ``obs_*`` 列）时不崩，且跳过条件覆盖图；
  4. ``rollout``/``fc`` 未启用时跳过对应图，其余照常；
  5. 分组表与前缀解析口径：按 ``subject_id`` 首段成组，>1.5×中位数的编组被标出；
  6. ``_pearson`` 与已知答案一致，且对缺失/常量序列返回 nan 而不是 0；
  7. 病理分层：五类分层器都落进 ``pathology_strata.csv``，每层被试数不超过总人数
     （防标签广播重复计数），摘要行带 Spearman/Kruskal-Wallis 读数；
  8. 临床列缺失（无 illness/HAMA）时出图不崩，且不产出该分层的摘要行。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import evaluation_report as ter

pytest.importorskip('matplotlib')


def _metrics(*, rollout: bool = True, fc: bool = True) -> dict:
    t1 = {'PCC_spatial': .99, 'PCC_spatial_std': .002, 'MAE': .074, 'RMSE': .092,
          'R2': .986, 'delta_direction': .95}
    m = {'protocol': {'n_subjects': 10, 'n_tasks': 12, 'prediction_target': 'delta_residual'},
         'next_state': {'model': {'t+1': t1}},
         'baselines': {'comparison': {
             'delta_MAE_model_minus_persistence': {'t+1': -.45}}},
         'hamd': {'corr_hamd_PCC_t+1': -.03},
         'rollout': {'enabled': rollout, 'model': {}, 'baselines': {},
                     'improvement_vs_persistence': {}},
         'cpm_horizon': {'enabled': rollout, 'model': {}},
         'fc': {'model': {'available': fc}},
         'spectral': {'enabled': False}}
    if rollout:
        hs = ['H1', 'H2']
        curve = {h: {'PCC_spatial': .99 - i * .05, 'MAE': .07 + i * .1,
                     'variance_ratio': .96 - i * .1} for i, h in enumerate(hs)}
        m['rollout']['model'] = curve
        m['rollout']['baselines'] = {'persistence': {h: dict(curve[h]) for h in hs},
                                     'trend': {h: dict(curve[h]) for h in hs}}
        m['rollout']['improvement_vs_persistence'] = {
            h: {'delta_PCC': .2, 'delta_MAE': -.4} for h in hs}
        m['cpm_horizon']['model'] = {'H1': curve['H1'], 'H2': curve['H2']}
    if fc:
        m['fc'] = {'model': {'available': True, 'n_edges': 10, 'length': 32,
                             'fc_mae': .35, 'edge_pcc_mean': .48,
                             'network': {'within': {'n_edges': 4, 'fc_mae': .34,
                                                     'edge_pcc_mean': .5},
                                         'between': {'n_edges': 6, 'fc_mae': .35,
                                                     'edge_pcc_mean': .47}}},
                   'persistence': {'available': True, 'fc_mae': .37,
                                   'edge_pcc_mean': .0}}
    return m


def _subject_df(condition: bool) -> pd.DataFrame:
    df = pd.DataFrame({
        'subject_id': ['S14-1-0001', 'IS012-1-0002', 'S20-1-0003',
                       'S20-1-0004', 'IS001-1-0005', 'IS001-1-0006',
                       'S1-1-0007', 'S2-1-0008', 'S3-1-0009', 'S5-1-0010'],
        'hamd': [31., 22., 18., 25., 20., 15., 24., 19., 27., 21.],
        't+1_MAE': [.12, .11, .07, .068, .066, .065, .067, .069, .070, .066],
        't+1_RMSE': [.15] * 10,
        't+1_PCC_spatial': [.983, .986, .994, .995, .995, .996, .994, .995, .993, .996],
        't+1_R2': [.98] * 10,
        'persistence_t+1_MAE': [.66, .65, .52, .51, .50, .49, .50, .51, .52, .49],
        'trend_t+1_MAE': [.60, .59, .46, .45, .44, .43, .44, .45, .46, .43],
        'ar1_t+1_MAE': [.62, .61, .48, .47, .46, .45, .46, .47, .48, .45]})
    if condition:
        for f, vals in {'HAMD': df['hamd'].to_numpy(),
                        'Sex': [1, 2, 1, 1, 2, 1, 2, 1, 1, 2], 'Age': [30, 40, 25, 33, 28, 45, 31, 29, 37, 42],
                        'HAMA': [0, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                        'episode': [0, 1, 2, 0, 1, 0, 2, 1, 0, 2],
                        'medication': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        'illness': [4, 12, 30, 8, 60, 100, 2, 16, 44, 24]}.items():
            df[f'cond_{f}'] = vals
            df[f'obs_{f}'] = 1
        df.loc[2, 'obs_HAMA'] = 0            # 让 HAMA 只有部分被试可见
        df.loc[2, 'obs_illness'] = 0
    return df


def _task_df() -> pd.DataFrame:
    return pd.DataFrame({'subj_id': np.repeat(['S14-1-0001', 'IS012-1-0002'], 6),
                         'context_len': [64] * 12,
                         'cutoff': list(range(63, 69)) * 2,
                         't+1_MAE': np.linspace(.06, .13, 12),
                         't+1_RMSE': np.linspace(.08, .16, 12),
                         't+1_PCC_spatial': np.linspace(.996, .982, 12),
                         't+1_R2': np.linspace(.99, .97, 12)})


def _write_eval_dir(tmp: Path, *, condition: bool = True) -> Path:
    d = tmp / 'eval'
    d.mkdir(parents=True, exist_ok=True)
    (d / '指标_测试集.json').write_text(json.dumps(_metrics()), encoding='utf-8')
    _subject_df(condition).to_csv(d / '逐被试指标_测试集.csv', index=False)
    _task_df().to_csv(d / '逐任务指标_测试集.csv', index=False)
    return d


def test_full_run_writes_all_figures(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    subj, task = _subject_df(True), _task_df()
    metrics = _metrics()
    ter.plot_vs_baselines(subj, task, out)
    ter.plot_subject_profile(subj, metrics, out)
    ter.plot_horizon(metrics, out)
    ter.plot_fc(metrics, out)
    ter.plot_condition_coverage(subj, out)
    ter.plot_task_structure(task, out)
    ter.plot_group_effect(subj, out)
    ter.plot_pathology(subj, out)
    names = sorted(p.name for p in out.glob('*.png'))
    assert names == ['fig1_next_state_vs_baselines.png', 'fig2_subject_profile.png',
                     'fig3_horizon_degradation.png', 'fig4_functional_connectivity.png',
                     'fig5_condition_coverage.png', 'fig6_task_structure.png',
                     'fig7_group_effect.png', 'fig8_pathology_strata.png']
    for p in out.glob('*.png'):
        assert p.stat().st_size > 10_000, p


def _many_subject_df() -> pd.DataFrame:
    """把 10 人模板平铺成 60 人，让 illness 三分位与 HAMA 共病面板走「有数据」分支。"""
    base = _subject_df(True)
    parts = [base.assign(subject_id=[f'{s}-rep{k}' for s in base['subject_id']])
             for k in range(6)]
    return pd.concat(parts, ignore_index=True)


def test_pathology_writes_strata_table(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    subj = _many_subject_df()
    rows = dict(ter.plot_pathology(subj, out, hot_groups=('S14', 'IS012')))  # noqa: E999
    tab = pd.read_csv(out / 'pathology_strata.csv')
    assert {'stratifier', 'strata', 'n', 'mae_mean', 'persistence_mean',
            'gain_mean', 'kruskal_p'} <= set(tab.columns)
    assert set(tab['stratifier']) >= {'HAMD_quintile', 'episode', 'medication',
                                      'Sex', 'illness_tertile'}
    assert tab['n'].sum() > 0
    assert (out / 'fig8_pathology_strata.png').exists()
    assert any(k.startswith('rho(MAE, HAMD)') for k in rows)
    assert any(k.startswith('illness_tertile:') for k in rows)
    # 分层表里的被试数不会超过总人数（防止标签广播造成重复计数）
    for name, g in tab.groupby('stratifier'):
        assert g['n'].sum() <= len(subj), name


def test_pathology_handles_missing_clinical_columns(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    subj = _subject_df(True).drop(columns=['cond_illness', 'obs_illness',
                                           'cond_HAMA', 'obs_HAMA'])
    rows = dict(ter.plot_pathology(subj, out))
    assert (out / 'fig8_pathology_strata.png').exists()
    assert not any(k.startswith('illness_tertile:') for k in rows)


def test_missing_artifact_raises_with_path(tmp_path):
    d = _write_eval_dir(tmp_path)
    (d / '逐任务指标_测试集.csv').unlink()
    with pytest.raises(FileNotFoundError, match='逐任务指标_测试集.csv'):
        ter.load_eval_artifacts(d, 'test')


def test_required_column_missing_raises(tmp_path):
    d = _write_eval_dir(tmp_path)
    df = _subject_df(True).drop(columns=['t+1_MAE'])
    df.to_csv(d / '逐被试指标_测试集.csv', index=False)
    with pytest.raises(ValueError, match='t\\+1_MAE'):
        ter.load_eval_artifacts(d, 'test')


def test_conditionless_run_skips_condition_figure(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    subj = _subject_df(False)
    ter.plot_condition_coverage(subj, out)
    assert not (out / 'fig5_condition_coverage.png').exists()
    ter.plot_subject_profile(subj, _metrics(), out)          # 不崩
    ter.plot_group_effect(subj, out)


def test_disabled_rollout_and_fc_skip_their_figures(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    ter.plot_horizon(_metrics(rollout=False), out)
    ter.plot_fc(_metrics(fc=False), out)
    assert not (out / 'fig3_horizon_degradation.png').exists()
    assert not (out / 'fig4_functional_connectivity.png').exists()


def test_group_table_flags_only_outlier_prefix(tmp_path):
    g = ter._group_table(_subject_df(True))
    assert set(g.index) == {'S14', 'IS012', 'S20', 'IS001', 'S1', 'S2', 'S3', 'S5'}
    hot = g[g['mae'] > 1.5 * g['mae'].median()]
    assert set(hot.index) == {'S14', 'IS012'}
    assert g.loc['S20', 'n'] == 2


def test_summary_rows_include_group_finding():
    rows = dict(ter.build_summary(_metrics(), _subject_df(True), _task_df(),
                                  ter._group_table(_subject_df(True))))
    assert 'IS012' in rows['>1.5x-median groups']
    assert rows['t+1 spatial PCC'].startswith('0.9900')


def test_pearson_matches_known_values():
    x = np.array([1., 2., 3., 4.])
    assert ter._pearson(x, 2 * x + 1) == pytest.approx(1.0)
    assert ter._pearson(x, -x) == pytest.approx(-1.0)
    assert np.isnan(ter._pearson(x, np.array([1., 1., 1., 1.])))
    assert np.isnan(ter._pearson(np.array([1., np.nan]), np.array([1., 2.])))
    assert ter._spearman(x, np.array([10., 20., 30., 41.])) == pytest.approx(1.0)
