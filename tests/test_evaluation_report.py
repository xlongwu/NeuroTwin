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
