"""ROI展示读取当前H32评价结构，不把描述性ΔMAE当作显著性检验。"""
import numpy as np
import pandas as pd

from analysis.visualize_roi_importance import prepare_data, generate_summary_table


def test_current_roi_effects_keep_sign_and_do_not_invent_fdr(tmp_path):
    report = {'roi_importance': {'mae_increase': [.2, -.1, .05]}}
    atlas = pd.DataFrame({'micro命名': ['A', 'B', 'C'], '对应网络': ['VN', 'DMN', 'VN']})
    frame, ranked = prepare_data(report, atlas)
    np.testing.assert_array_equal(frame.Importance, [.2, -.1, .05])
    assert ranked.ROI_Index.tolist() == [0, 2, 1]
    result = generate_summary_table(ranked, pd.DataFrame(), tmp_path / '汇总.json')
    assert 'n_significant_fdr' not in result['overall_stats']
