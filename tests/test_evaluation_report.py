"""正式 H32 报告契约；不依赖历史双头结果或绘图环境。"""
import json

import pandas as pd
import pytest

from analysis.evaluation_report import load_eval_artifacts, write_report


def _write(directory):
    directory.mkdir()
    tables = directory / '表格'
    tables.mkdir()
    trajectory = {key: {'H1_32': {'mean': value}} for key, value in
                  dict(mae=.8, rmse=1., temporal_pcc=.46, fc_mse=.05, freq_l1=.1).items()}
    metrics = dict(protocol=dict(model='CPM', horizon=32, n_subjects=2, n_anchors=4),
                   trajectory_metrics=trajectory,
                   baselines={'persistence': {'trajectory_metrics': trajectory}})
    (directory / '指标_测试集.json').write_text(json.dumps(metrics))
    frame = pd.DataFrame([dict(模型='CPM', 预测范围='H1_32', mae=.8, rmse=1., 被试='S1')])
    for name in ('逐时距指标', '分阶段指标', '逐被试分阶段指标'):
        frame.to_csv(tables / f'{name}_测试集.csv', index=False)
    return directory


def test_current_report_reads_and_exports(tmp_path):
    source = _write(tmp_path / 'eval')
    data = write_report(source, tmp_path / '报告')
    assert data['metrics']['protocol']['horizon'] == 32
    assert '0.460000' in (tmp_path / '报告/评价说明.md').read_text()
    comparison = pd.read_csv(tmp_path / '报告/表格/完整时距模型与基线.csv')
    assert comparison['模型'].tolist() == ['CPM', 'persistence']
    assert len(list((tmp_path / '报告/表格').glob('*.csv'))) == 4


def test_required_artifacts_and_columns(tmp_path):
    source = _write(tmp_path / 'eval')
    path = source / '表格/逐时距指标_测试集.csv'
    pd.DataFrame([dict(模型='CPM')]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='缺少列'):
        load_eval_artifacts(source)
    path.unlink()
    with pytest.raises(FileNotFoundError, match='逐时距指标'):
        load_eval_artifacts(source)


def test_rejects_retired_report_protocol(tmp_path):
    source = _write(tmp_path / 'eval')
    path = source / '指标_测试集.json'
    payload = json.loads(path.read_text())
    payload['protocol']['model'] = 'ODE'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='CPM H32'):
        load_eval_artifacts(source)


def test_same_protocol_comparison_exports_four_pdf_figures(tmp_path, monkeypatch):
    """仅工程用合成fixture；不属于正式预测结果。"""
    from types import SimpleNamespace
    import numpy as np
    from analysis import next_point_eval as npe
    from analysis.evaluation_report import write_comparison
    from matplotlib.figure import Figure
    original_title = Figure.suptitle
    monkeypatch.setattr(Figure, 'suptitle', lambda self, text, *a, **kw:
                        original_title(self, '工程测试（合成数据）｜'+text, *a, **kw))
    rng = np.random.default_rng(2024)
    target = rng.normal(size=(4, 4, 32)).astype('float32')
    history = rng.normal(size=(4, 4, 64)).astype('float32')
    data = dict(target=target, history=history, subjects=np.array(['S1', 'S1', 'S2', 'S2']),
                cutoff=np.array([63, 79, 63, 79]), anchor=history[..., -1], previous=history[..., -2],
                mask=np.ones((4, 32), dtype='float32'), history_mean=history.mean(-1))
    args = SimpleNamespace(cpm_horizon=32, eval_context_length=64, mode='pretrain', calibration_length=32)
    for name, noise, enabled in [('original', .4, False), ('optimized', .2, True)]:
        args.multiscale_phase = enabled
        current = {**data, 'pred': target+rng.normal(scale=noise, size=target.shape).astype('float32')}
        monkeypatch.setattr(npe, 'run_inference', lambda *a, _data=current, **kw: _data)
        npe.evaluate_split(None, None, args, 'cpu', tmp_path/name, 'test', save_arrays=True,
                           split_identity='same-split', checkpoint_identity={'test_fixture': True})
    out = tmp_path/'comparison'
    write_comparison(tmp_path/'original', tmp_path/'optimized', out)
    assert len(list((out/'图片').glob('*.pdf'))) == 4
    assert (out/'说明/模型比较.md').is_file()
    frame = pd.read_csv(out/'表格/分阶段指标对比.csv')
    assert 'delta_pcc' in frame and 'log_psd_mae' in frame and 'fc_pcc' in frame
    with np.load(tmp_path/'optimized/预测数组_测试集.npz') as archive:
        corrupted = {key: archive[key] for key in archive.files}
    corrupted['history'] = corrupted['history'].copy()
    corrupted['history'][0, 0, 0] += 1
    np.savez(tmp_path/'optimized/预测数组_测试集.npz', **corrupted)
    with pytest.raises(ValueError, match='history不一致'):
        write_comparison(tmp_path/'original', tmp_path/'optimized', tmp_path/'mismatch')


def test_experiment_summary_keeps_stage_gains_and_individual_horizon_regressions(tmp_path, monkeypatch):
    from analysis.evaluation_report import write_experiment_summary
    from train import training_state
    monkeypatch.setattr(training_state, 'read_training_state_metadata', lambda path: {
        'progress': {'epoch': 22, 'early_stopped': True, 'backbone_unfrozen': True}})
    stages = ('H1_8', 'H9_16', 'H17_32', 'H1_32')
    primary = dict(mae=.8, rmse=1., temporal_pcc=.4)
    dynamic = dict(std_ratio=.5, delta_pcc=.3, delta_mae=.4, psd_shape_pcc=.6,
                   log_psd_mae=1., fc_mse=.2, fc_pcc=.5)
    for mode, old_name, new_name in [('pretrain', '原B健康评价', '新模型健康评价'),
                                      ('finetune', '原B抑郁症评价', '新模型抑郁症评价')]:
        run = tmp_path/'runs'/mode
        run.mkdir(parents=True)
        (run/'验证训练记录.jsonl').write_text('\n'.join(json.dumps(row) for row in [
            {'epoch': 10, 'val': {'metric_selection_h32': .7}},
            {'epoch': 22, 'val': {'metric_selection_h32': .8}}]))
        for enabled, name in [(False, old_name), (True, new_name)]:
            directory=tmp_path/name;tables=directory/'表格';tables.mkdir(parents=True)
            values={**primary, **dynamic}
            if enabled:values.update(mae=.64, rmse=.8, temporal_pcc=.5, std_ratio=.45)
            protocol=dict(model='CPM',horizon=32,context_length=64,tr_seconds=2.,calibration_length=32,
                          normalization='same',mode=mode,split_hash=mode,n_subjects=2,n_anchors=4,
                          multiscale_phase=enabled,checkpoint={'path':str(run/'best.pt'),'sha256':'fixture'})
            payload=dict(protocol=protocol,trajectory_metrics={key:{s:{'mean':v} for s in stages}
                                                               for key,v in values.items()})
            (directory/'指标_测试集.json').write_text(json.dumps(payload))
            frame=pd.DataFrame([dict(模型='CPM',预测范围=f'H{i}',mae=.64 if enabled else .8,
                                     rmse=.8 if enabled else 1.,spatial_pcc=.6 if enabled else .5)
                                for i in range(1,33)])
            if enabled and mode=='finetune':frame.loc[19,'mae']=.81
            frame.to_csv(tables/'逐时距指标_测试集.csv',index=False)
            stage=pd.DataFrame([dict(模型='CPM',预测范围=s,mae=values['mae'],rmse=values['rmse'],被试='S1') for s in stages])
            for table in ('分阶段指标','逐被试分阶段指标'):
                stage.to_csv(tables/f'{table}_测试集.csv',index=False)
    frames=write_experiment_summary(tmp_path,tmp_path/'汇总')
    assert len(frames['分阶段主指标汇总'])==8 and len(frames['逐时距变化汇总'])==64
    assert frames['分阶段主指标汇总']['mae_下降百分比'].tolist()==pytest.approx([20.]*8)
    report=(tmp_path/'汇总/说明/实施结果总结.md').read_text()
    assert '31/32' in report and 'H20' in report and '幅值收缩没有缓解' in report
