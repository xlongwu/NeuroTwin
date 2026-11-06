"""Covariate-adjusted exploratory cohort tests for frozen-model mechanism responses."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from utils.condition_schema import file_sha256 as digest
from analysis.pathology_tests import bh_fdr
from analysis.shared_mechanism import GROUP_FILES, NETWORKS, ROOT, network_responses, outgoing_response
from utils.dataloader import load_condition_table
from utils.site_subset import parse_site

ARMS = ('combined', 'fixed_graph', 'graph_contrast', 'fc_baseline')


def nuisance_design(age: np.ndarray, sex: np.ndarray, sites: np.ndarray) -> np.ndarray:
    """Use independent nuisance columns; single-sex subsets need no sex dummy."""
    columns = [np.ones(len(age))]
    if np.std(age) > 0:
        columns.append((age-age.mean())/age.std())
    for value in sorted(set(sex))[1:]:
        columns.append((sex == value).astype(float))
    for value in sorted(set(sites))[1:]:
        columns.append((sites == value).astype(float))
    design = np.column_stack(columns)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError('Age/sex/site nuisance design is rank deficient')
    return design


def adjusted_group_test(y: np.ndarray, group: np.ndarray, nuisance: np.ndarray,
                        sites: np.ndarray, n_perm: int = 10000) -> dict:
    """OLS coefficient, HC3 interval, and within-site Freedman–Lane residual permutation."""
    y = np.asarray(y, dtype=float)
    residual = y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]
    rg = group - nuisance @ np.linalg.lstsq(nuisance, group, rcond=None)[0]
    if rg @ rg <= 1e-12:
        raise ValueError('Group is not identifiable after covariate adjustment')
    weights = rg / (rg @ rg)
    beta = weights @ y
    full = np.column_stack([nuisance, group])
    inv = np.linalg.pinv(full)
    error = y - full @ (inv @ y)
    leverage = np.sum(full * inv.T, axis=1)
    if (leverage >= 1-1e-10).any():
        raise ValueError('HC3 undefined for a leverage-one observation')
    se = np.sqrt(np.sum((weights[:, None] * error / (1-leverage[:, None]))**2, axis=0))
    critical = stats.t.ppf(.975, len(y)-np.linalg.matrix_rank(full))
    rng = np.random.default_rng(2024)
    by_site = [np.flatnonzero(sites == s) for s in sorted(set(sites))]
    exceed = np.zeros(y.shape[1], dtype=np.int64)
    for start in range(0, n_perm, 200):
        count = min(200, n_perm-start)
        indices = np.tile(np.arange(len(y)), (count, 1))
        for ix in by_site:
            indices[:, ix] = np.array([rng.permutation(ix) for _ in range(count)])
        null = np.einsum('i,bim->bm', weights, residual[indices])
        exceed += (np.abs(null) >= np.abs(beta)[None, :]-1e-12).sum(axis=0)
    scale = np.sqrt(np.sum(residual**2, axis=0)/(len(y)-np.linalg.matrix_rank(nuisance)))
    return dict(beta=beta, ci_low=beta-critical*se, ci_high=beta+critical*se,
                standardized_beta=np.divide(beta, scale, out=np.full_like(beta, np.nan), where=scale > 0),
                p=(exceed+1)/(n_perm+1))


def partial_correlation(y: np.ndarray, x: np.ndarray, nuisance: np.ndarray) -> tuple:
    rx = x - nuisance @ np.linalg.lstsq(nuisance, x, rcond=None)[0]
    ry = y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]
    dof = len(x)-np.linalg.matrix_rank(nuisance)-1
    if dof <= 0 or np.std(rx) <= 1e-12 or (np.std(ry, axis=0) <= 1e-12).any():
        raise ValueError('Clinical partial correlation has insufficient degrees of freedom or zero variance')
    r = rx @ ry / np.sqrt((rx @ rx)*np.sum(ry**2, axis=0))
    r = np.clip(r, -1, 1)
    t = r*np.sqrt(dof/np.maximum(1-r*r, 1e-15))
    return r, 2*stats.t.sf(abs(t), dof), dof


def load_cohort(cache: Path) -> tuple:
    atlas = pd.read_excel(ROOT/'data/AAL116.xlsx')['对应网络'].to_numpy()
    manifests, records, values, blocks = {}, [], [], []
    for group, (split, condfile) in GROUP_FILES.items():
        directory = cache/group
        manifest = json.loads((directory/'数据清单.json').read_text())
        expected = json.loads((ROOT/'data'/split).read_text())['test']
        if not manifest['complete_test'] or manifest['subjects'] != expected:
            raise ValueError(f'{group}: complete original test split required')
        if manifest['split_sha256'] != digest(ROOT/'data'/split) or manifest['atlas_sha256'] != digest(ROOT/'data/AAL116.xlsx'):
            raise ValueError(f'{group}: split or atlas changed')
        if manifest['condition_table_sha256'] != digest(ROOT/'data'/condfile):
            raise ValueError(f'{group}: condition table changed')
        if group == 'mdd':
            shared_keys = ('version', 'checkpoint_sha256', 'source_sha256', 'seed', 'repeats', 'anchors',
                           'permutation', 'arms', 'calibration', 'primary', 'graph_contrast', 'condition_group', 'torch')
            if any(manifest[k] != manifests['hc'][k] for k in shared_keys):
                raise ValueError('Cohorts do not share one model and analysis protocol')
        manifests[group] = manifest
        table = load_condition_table(str(ROOT/'data'), condfile, 'hc3')
        columns = {s.name:i for i,s in enumerate(table.specs)}
        demo, mask = table.align(expected)
        for i, sid in enumerate(expected):
            with np.load(directory/f'{sid}.npz') as z:
                base, combined, fixed = z['baseline'], z['combined'], z['fixed_graph']
                if (combined.shape != (2,116,116) or fixed.shape != combined.shape or base.shape != (2,116)
                        or z['cutoff'][1]-z['cutoff'][0] < 64
                        or float(z['sham_max']) > 1e-5 or float(z['fixed_graph_max_diff']) != 0):
                    raise ValueError(f'{sid}: invalid cache or failed graph checks')
                responses = [abs(combined-base[:,None,:]), abs(fixed-base[:,None,:]),
                             abs(combined-fixed), abs(z['perturbed_fc_rows']-z['history_fc'])]
                if not all(np.isfinite(r).all() for r in responses):
                    raise ValueError(f'{sid}: nonfinite response')
                values.append(np.array([[outgoing_response(window, atlas) for window in r] for r in responses]))
                blocks.append(np.array([[network_responses(window, atlas) for window in r] for r in responses]))
                records.append(dict(subject=sid, group=group, site=parse_site(sid),
                                    age=float(demo[i,columns['Age']]) if mask[i,columns['Age']] else np.nan,
                                    sex=float(demo[i,columns['Sex']]) if mask[i,columns['Sex']] else np.nan,
                                    sham_max=float(z['sham_max']), graph_change=float(z['recomputed_graph_max_diff'])))
    return pd.DataFrame(records), np.array(values), np.array(blocks), manifests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    dest = a.out_dir
    dest.mkdir(parents=True, exist_ok=True)
    subjects, windows, blocks, manifests = load_cohort(a.cache_dir)
    subjects.to_csv(dest/'被试与质控.csv', index=False)
    rows = []
    for i, row in subjects.iterrows():
        for arm, name in enumerate(ARMS):
            for net, network in enumerate(NETWORKS):
                rows.append(dict(subject=row.subject, group=row.group, arm=name, network=network,
                                 early=windows[i,arm,0,net], late=windows[i,arm,1,net],
                                 mean=windows[i,arm,:,net].mean()))
    pd.DataFrame(rows).to_csv(dest/'逐被试网络响应.csv', index=False)
    np.savez_compressed(dest/'网络块响应.npz', subjects=subjects.subject.to_numpy(dtype=str),
                        arms=np.array(ARMS), networks=np.array(NETWORKS), response=blocks)
    eligible = subjects[subjects.age.notna() & subjects.sex.notna()]
    shared = set(eligible[eligible.group=='hc'].site) & set(eligible[eligible.group=='mdd'].site)
    keep = subjects.age.notna() & subjects.sex.notna() & subjects.site.isin(shared)
    cohort = subjects[keep]
    y = windows[keep].mean(axis=2).reshape(len(cohort), -1)
    nuisance = nuisance_design(cohort.age.to_numpy(), cohort.sex.to_numpy(), cohort.site.to_numpy())
    g = (cohort.group=='mdd').to_numpy(dtype=float)
    result = adjusted_group_test(y, g, nuisance, cohort.site.to_numpy())
    # Primary 7; secondary 14; simple FC reference 7. Never choose families after seeing results.
    q = np.concatenate([bh_fdr(result['p'][:7]), bh_fdr(result['p'][7:21]), bh_fdr(result['p'][21:])])
    group_rows = []
    for j in range(28):
        group_rows.append(dict(arm=ARMS[j//7], network=NETWORKS[j%7],
                               hc_mean=float(y[g==0,j].mean()), mdd_mean=float(y[g==1,j].mean()),
                               **{k:float(v[j]) for k,v in result.items()}, q=float(q[j]),
                               n_hc=int(sum(g==0)), n_mdd=int(sum(g==1))))
    groups = pd.DataFrame(group_rows)
    groups.to_csv(dest/'组间检验.csv', index=False)
    stability = []
    for group in GROUP_FILES:
        ix = (subjects.group == group).to_numpy()
        valid = ix & subjects.age.notna().to_numpy() & subjects.sex.notna().to_numpy()
        sub = subjects[valid]
        c = nuisance_design(sub.age.to_numpy(), sub.sex.to_numpy(), sub.site.to_numpy())
        adjusted = windows[valid,0].reshape(len(sub), -1)
        adjusted -= c @ np.linalg.lstsq(c, adjusted, rcond=None)[0]
        adjusted = adjusted.reshape(len(sub),2,7)
        for j, network in enumerate(NETWORKS):
            early, late = windows[ix,0,0,j], windows[ix,0,1,j]
            r = float(stats.pearsonr(early, late).statistic)
            ar = float(stats.pearsonr(adjusted[:,0,j], adjusted[:,1,j]).statistic)
            stability.append(dict(group=group, network=network, n=int(ix.sum()), window_pearson=r,
                                  n_adjusted=len(sub), adjusted_window_pearson=ar))
    stable = pd.DataFrame(stability)
    stable.to_csv(dest/'窗口稳定性.csv', index=False)
    rank = subjects[['subject','group']].copy()
    rank['network_rank_spearman'] = [stats.spearmanr(w[0,0], w[0,1]).statistic for w in windows]
    rank.to_csv(dest/'个体窗口排序.csv', index=False)
    mdd = subjects[subjects.group=='mdd']
    table = load_condition_table(str(ROOT/'data'), GROUP_FILES['mdd'][1], 'mdd8')
    clinical, mask = table.align(mdd.subject.tolist())
    cols = {s.name:i for i,s in enumerate(table.specs)}
    clinical_rows = []
    for score in ('HAMD', 'HAMA'):
        ok = (mask[:,cols[score]].astype(bool) & mdd.age.notna().to_numpy() & mdd.sex.notna().to_numpy())
        sub = mdd[ok]
        c = nuisance_design(sub.age.to_numpy(), sub.sex.to_numpy(), sub.site.to_numpy())
        r, prob, dof = partial_correlation(windows[subjects.group=='mdd',0].mean(axis=1)[ok], clinical[ok,cols[score]], c)
        for j, network in enumerate(NETWORKS):
            clinical_rows.append(dict(score=score, network=network, n=int(ok.sum()), n_excluded=int((~ok).sum()),
                                      dof=dof, partial_r=float(r[j]), p=float(prob[j])))
    clinic = pd.DataFrame(clinical_rows)
    clinic['q'] = bh_fdr(clinic.p.to_numpy())
    clinic.to_csv(dest/'临床关联.csv', index=False)
    summary = dict(n_total=len(subjects), n_group_analysis=len(cohort), shared_sites=len(shared),
                   excluded_missing_demo=int((subjects.age.isna()|subjects.sex.isna()).sum()),
                   excluded_nonshared_site=int((subjects.age.notna() & subjects.sex.notna() & ~subjects.site.isin(shared)).sum()),
                   n_hc=int(sum(g==0)), n_mdd=int(sum(g==1)),
                   significant_by_arm={arm:int(((groups.arm==arm)&(groups.q<.05)).sum()) for arm in ARMS},
                   clinical_significant=int((clinic.q<.05).sum()),
                   median_window_pearson={group:float(stable[stable.group==group].window_pearson.median()) for group in GROUP_FILES},
                   median_adjusted_window_pearson={group:float(stable[stable.group==group].adjusted_window_pearson.median()) for group in GROUP_FILES},
                   max_sham=float(subjects.sham_max.max()), n_perm=10000, seed=2024,
                   checkpoint_sha256=manifests['hc']['checkpoint_sha256'],
                   inference_source_sha256=manifests['hc']['source_sha256'], summary_source_sha256=digest(Path(__file__)),
                   limitations=['explored test set; exploratory only', 'same scan and shared encoder; no retest validation',
                                'site residual exchangeability assumed; HC3 is individual-level, not site-cluster interval',
                                'single common permutation seed and one permutation per ROI',
                                'graph contrast is conditional sensitivity, not neural causality or additive mediation',
                                'FC baseline comparison does not establish predictive increment'])
    (dest/'结果汇总.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 同模型病理响应分析结果', '',
             '固定 HC 模型分析两组测试集；测试集此前已经探索，本次结果不属于独立确认。', '',
             f'完整推理 {len(subjects)} 人；组间检验纳入 HC {sum(g==0)} 人、MDD {sum(g==1)} 人，共 {len(shared)} 个共有站点。', '',
             '| 分析 | FDR 显著网络数 |', '|---|---:|']
    lines += [f'| {arm} | {summary["significant_by_arm"][arm]}/7 |' for arm in ARMS]
    lines += ['', f'症状关联经 14 项统一 FDR 后显著 {summary["clinical_significant"]}/14 项。HAMD 有效 {int(clinic[clinic.score=="HAMD"].n.iloc[0])} 人，HAMA 有效 {int(clinic[clinic.score=="HAMA"].n.iloc[0])} 人；HAMA 缺失较多，相关阴性结果的解释受限。', '',
              '| 源网络 | MDD 减 HC 调整系数 | HC3 95% 区间 | 标准化系数 | q |', '|---|---:|---|---:|---:|']
    for row in group_rows[:7]:
        lines.append(f'| {row["network"]} | {row["beta"]:.6g} | {row["ci_low"]:.6g} 至 {row["ci_high"]:.6g} | {row["standardized_beta"]:.3f} | {row["q"]:.4f} |')
    lines += ['', f'七网络窗口间跨人 Pearson 相关的中位数：HC {summary["median_window_pearson"]["hc"]:.3f}，MDD {summary["median_window_pearson"]["mdd"]:.3f}；控制年龄、性别、站点后分别为 {summary["median_adjusted_window_pearson"]["hc"]:.3f}、{summary["median_adjusted_window_pearson"]["mdd"]:.3f}。这是同一扫描两个不重叠窗口的稳定性，不是跨扫描重测。', '',
              'combined 为正常重算图响应；fixed_graph 为固定原有效图响应；graph_contrast 为两者预测差的绝对值；fc_baseline 为相同扰动下的历史 FC 变化。均排除 ROI 自环。', '',
              '主终点 7 项单独 FDR；两类次终点共 14 项 FDR；FC 基线 7 项单独 FDR。组间效应控制年龄、性别和站点，使用 10000 次站内残差置换。', '',
              '图重算对照可定位模型敏感性的来源，但不能推断真实神经因果。一个种子的扰动不能概括全部扰动；绝对输出响应与 FC 响应量纲不同，不直接比较原值。显著项数量不代表模型超越基线。', '',
              '完整效应见 [组间检验](组间检验.csv)，症状关联见 [临床关联](临床关联.csv)，稳定性见 [窗口稳定性](窗口稳定性.csv)，输入和代码哈希见缓存清单及 [结果汇总](结果汇总.json)。']
    if summary['significant_by_arm']['combined'] == 0 and summary['clinical_significant'] == 0:
        lines += ['', '## 研究判读', '',
                  '在固定同一个 HC 模型、统一条件编码、排除自身响应和控制年龄、性别、站点后，本次跨 ROI 响应分析仍未建立稳定的 MDD–HC 疾病相关差异。组间阴性不证明两组等价，也不能排除其他特征、时程或患者子群中的疾病信息。', '',
                  '未经调整的窗口相关较高，不能独立作为个体特异性的依据。调整后相关下降提示，与这些协变量共享的变异对原始稳定性有重要贡献；该比较不能单独证明站点的因果作用，也没有检验两组稳定性的差异是否显著。', '',
                  '若目标是稳定的个体病理标志，下一步优先应解决响应表征在控制混杂后的跨窗口复现问题。低相关也可能源于脑状态变化、有限窗口或置换探针的扰动方式，因此现有结果不足以指定唯一模型结构缺陷，亦不足以要求直接改造模型。', '',
                  '本次未执行跨站点预测增量或独立外部验证，没有确立模型优于 FC 基线。结果不支持刺激、治疗或疾病因果结论。']
    (dest/'分析报告.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
