"""只读分析固定训练名单的 BOLD；PDF 图、CSV 表和 JSON 记录分开保存。

python analysis/训练数据多尺度分解.py --data-root data \
  --output results/训练数据分析/时序多尺度分解_20261107 --tr 2

频谱/小波整段分解仅用于描述，绝不作为无泄漏预测结果。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import pywt
from scipy import signal
from scipy.integrate import trapezoid
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.h18_data import calibrate_raw
from utils.dataloader import read_mat_array


BANDS = [(0, .01), (.01, .027), (.027, .073), (.073, .1), (.1, .25)]
BAND_NAMES = ["低于0.01", "0.01–0.027", "0.027–0.073", "0.073–0.1", "高于等于0.1"]
COLORS = ["#6592ac", "#2b438d", "#93b5a1", "#db7c8d", "#d7b1c8"]
LAGS = [1, 8, 16, 32]


def corr_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(0)
    b = b - b.mean(0)
    den = np.sqrt((a*a).sum(0) * (b*b).sum(0))
    return np.divide((a*b).sum(0), den, out=np.full(den.shape, np.nan), where=den > 1e-16)


def fft_parts(x: np.ndarray, tr: float) -> list[np.ndarray]:
    """互斥 DFT 频点投影；重构精确但有整段/周期边界依赖。"""
    freq = np.fft.rfftfreq(len(x), tr)
    coeff = np.fft.rfft(x, axis=0)
    return [np.fft.irfft(coeff * ((freq >= lo) &
            ((freq < hi) if i < len(BANDS)-1 else (freq <= hi)))[:, None],
            n=len(x), axis=0) for i, (lo, hi) in enumerate(BANDS)]


def wave_parts(x: np.ndarray, wavelet: str = "db4") -> tuple[list[np.ndarray], list[np.ndarray]]:
    coeffs = pywt.wavedec(x, wavelet, level=3, mode="symmetric", axis=0)
    parts = []
    for i in range(len(coeffs)):
        selected = [c if j == i else np.zeros_like(c) for j, c in enumerate(coeffs)]
        parts.append(pywt.waverec(selected, wavelet, mode="symmetric", axis=0)[:len(x)])
    return parts, coeffs


def decomposition_metrics(x: np.ndarray, parts: list[np.ndarray]) -> dict:
    variance = np.var(x, axis=0)
    ratios = np.array([np.var(p, axis=0) / variance for p in parts])
    covariance = 1 - ratios.sum(0)
    return {"ratios": ratios, "cross": covariance,
            "reconstruction": float(np.max(np.abs(x - np.sum(parts, axis=0))))}


def record_parts(rows: list, base: dict, method: str, names: list[str],
                 x: np.ndarray, parts: list[np.ndarray]) -> float:
    m = decomposition_metrics(x, parts)
    for name, p, ratios in zip(names, parts, m["ratios"]):
        row = {**base, "方法": method, "成分": name,
               "ROI平均方差比": float(np.mean(ratios)),
               "ROI方差比中位数": float(np.median(ratios)),
               "交叉协方差占比": float(np.mean(m["cross"])),
               "重构最大绝对误差": m["reconstruction"]}
        for lag in LAGS:
            row[f"自相关{lag}TR"] = float(np.nanmean(corr_columns(p[:-lag], p[lag:])))
        if len(x) > 96:
            row["去边界24TR方差比"] = float(np.mean(np.var(p[24:-24], 0) / np.var(x[24:-24], 0)))
        rows.append(row)
    return m["reconstruction"]


def boundary_metrics(x: np.ndarray, base: dict, tr: float) -> list[dict]:
    """固定三锚点，分离未来依赖与64TR历史截断效应。"""
    full_wave, _ = wave_parts(x)
    full = {"趋势17TR": uniform_filter1d(x, 17, axis=0, mode="reflect"),
            "FFT低频0.01–0.027": fft_parts(x, tr)[1], "小波db4_A3": full_wave[0]}
    rows = []
    for end in np.unique(np.linspace(64, len(x)-32, 3).astype(int)):
        ref = {name: value[end-8:end] for name, value in full.items()}
        for scope, start in [("同一前缀无未来", 0), ("仅64TR历史", end-64)]:
            hist = x[start:end]
            wp, _ = wave_parts(hist)
            estimates = {"趋势17TR": uniform_filter1d(hist, 17, axis=0, mode="reflect"),
                         "FFT低频0.01–0.027": fft_parts(hist, tr)[1], "小波db4_A3": wp[0]}
            for name, est in estimates.items():
                rms = np.sqrt(np.mean((est[-8:]-ref[name])**2, axis=0)) / np.std(x, axis=0)
                rows.append({**base, "历史终点TR数": int(end), "可见范围": scope, "成分": name,
                             "末8TR差异除以原序列标准差": float(np.mean(rms))})
    return rows


def configure_plot() -> None:
    path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(path).exists():
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "pdf.fonttype": 3})


def figures(output: Path, spectra: pd.DataFrame, components: pd.DataFrame,
            boundaries: pd.DataFrame, examples: dict, tr: float) -> None:
    configure_plot()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for cohort, frame in spectra.groupby("队列"):
        label = cohort
        for version, sub in frame.groupby("版本"):
            ax = axes[0, 0]
            freqcols = [c for c in sub if c.startswith("频点_")]
            f = np.array([float(c[3:]) for c in freqcols])
            y = sub[freqcols].mean().to_numpy(float)
            ax.plot(f, y, label=f"{label} {version}")
    axes[0, 0].set(xlabel="频率（Hz）", ylabel="归一化 Welch PSD（1/Hz）", title="A  两批数据的频谱形状")
    axes[0, 0].legend(fontsize=8)
    for cut in [.01, .027, .073, .1]:
        axes[0, 0].axvline(cut, color="gray", lw=.6, alpha=.4)
    labels, values = [], []
    for (group, version), sub in components[components["方法"] == "FFT"].groupby(["队列", "版本"]):
        labels.append(f"{group}\n{version}")
        values.append([sub[sub["成分"] == n]["ROI平均方差比"].mean() for n in BAND_NAMES])
    bottom = np.zeros(len(labels))
    for k, name in enumerate(BAND_NAMES):
        v = np.array(values)[:, k]*100
        axes[0, 1].bar(labels, v, bottom=bottom, label=name, color=COLORS[k])
        for j, val in enumerate(v):
            if val >= 6:
                axes[0, 1].text(j, bottom[j]+val/2, f"{val:.1f}%", ha="center", va="center", fontsize=8,
                              color="white" if k == 1 else "black")
        bottom += v
    axes[0, 1].set(ylabel="方差占比（%）", title="B  互斥频带的能量分配", ylim=(0, 100))
    axes[0, 1].legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.15))
    for j, group in enumerate(["HC", "MDD"]):
        sub = components[(components["方法"] == "均值趋势33TR") & (components["成分"] == "趋势") & (components["队列"] == group)]
        for k, version in enumerate(["V1", "V2"]):
            vals = sub[sub["版本"] == version]["ROI平均方差比"]*100
            pos = j*3+k
            axes[1, 0].boxplot([vals], positions=[pos], widths=.6, showfliers=False,
                               patch_artist=True, boxprops={"facecolor": COLORS[j]}, medianprops={"color": "black"})
    axes[1, 0].set_xticks([0, 1, 3, 4], ["HC V1", "HC V2", "MDD V1", "MDD V2"])
    axes[1, 0].set(ylabel="趋势方差 / 原序列方差（%）", title="C  33TR慢成分在被试间的分布")
    for method, name, style in [("原序列", "原序列", "-"), ("均值趋势17TR", "趋势", "--"), ("均值趋势17TR", "残差", ":")]:
        sub = components[(components["方法"] == method) & (components["成分"] == name)]
        vals = [sub[f"自相关{lag}TR"].mean() for lag in LAGS]
        axes[1, 1].plot(np.array(LAGS)*tr, vals, style, marker="o", label=name)
    axes[1, 1].axhline(0, color="gray", lw=.7)
    axes[1, 1].set(xlabel="时距（秒）", ylabel="时间自相关（被试等权均值）", title="D  分解后的记忆尺度（整段诊断）")
    axes[1, 1].legend()
    fig.suptitle("训练集多尺度结构｜全部训练被试，先ROI平均再被试等权", fontsize=14)
    fig.savefig(output/"图片"/"训练数据多尺度结构总览.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for j, wavelet in enumerate(["db2", "db4"]):
        for k, group in enumerate(["HC", "MDD"]):
            sub = components[(components["方法"] == f"小波{wavelet}") & (components["队列"] == group)]
            names = ["A3", "D3", "D2", "D1"]
            vals = [sub[sub["成分"] == n]["ROI平均方差比"].mean()*100 for n in names]
            axes[0, j].bar(np.arange(4)+(k-.5)*.34, vals, width=.34, label=group, color=COLORS[k])
        axes[0, j].set_xticks(range(4), ["A3\n<0.03125", "D3\n0.03125–0.0625", "D2\n0.0625–0.125", "D1\n0.125–0.25"])
        axes[0, j].set(ylabel="重构成分方差 / 原序列方差（%）", title=f"{'AB'[j]}  {wavelet}三层分解（名义频带，Hz）")
        axes[0, j].legend()
    sub = boundaries.groupby(["被试", "队列", "可见范围", "成分"], as_index=False)["末8TR差异除以原序列标准差"].mean()
    for k, scope in enumerate(["同一前缀无未来", "仅64TR历史"]):
        means = sub[sub["可见范围"] == scope].groupby("成分")["末8TR差异除以原序列标准差"].mean()
        axes[1, 0].bar(np.arange(len(means))+(k-.5)*.34, means, width=.34, label=scope, color=COLORS[k])
    axes[1, 0].set_xticks(range(len(means)), ["FFT低频", "小波 A3", "17TR趋势"])
    axes[1, 0].set(ylabel="末8TR分量差异 / 原序列标准差", title="C  历史可见性改变后的边界偏差")
    axes[1, 0].legend(fontsize=8)
    names = ["A3", "D3", "D2", "D1"]
    for i, name in enumerate(names):
        sub = components[(components["方法"] == "小波db4") & (components["成分"] == name)]
        axes[1, 1].plot(np.array(LAGS)*tr, [sub[f"自相关{lag}TR"].mean() for lag in LAGS], marker="o", label=name, color=COLORS[i])
    axes[1, 1].axhline(0, color="gray", lw=.7)
    axes[1, 1].set(xlabel="时距（秒）", ylabel="时间自相关", title="D  不同小波尺度的记忆")
    axes[1, 1].legend()
    fig.suptitle("小波与边界诊断｜非因果分解不能直接证明可预测性", fontsize=14)
    fig.savefig(output/"图片"/"小波尺度与历史边界诊断.pdf")
    plt.close(fig)

    for (group, version), ex in examples.items():
        x, sid, roi = ex
        y = x[:, roi:roi+1]
        wp, _ = wave_parts(y)
        parts = fft_parts(y, tr)
        t = np.arange(len(y))*tr
        fig, axes = plt.subplots(4, 1, figsize=(12, 11), layout="constrained")
        trend = uniform_filter1d(y, 17, axis=0, mode="reflect")
        axes[0].plot(t, y[:, 0], color="#999999", lw=.8, label="原序列")
        axes[0].plot(t, trend[:, 0], color=COLORS[1], label="17TR趋势")
        axes[0].plot(t, (y-trend)[:, 0], color=COLORS[3], alpha=.65, lw=.8, label="残差")
        axes[0].legend(ncol=3, fontsize=8)
        axes[0].set(title="A  趋势与残差", ylabel="校准尺度")
        for i, (p, name) in enumerate(zip(parts, BAND_NAMES)):
            axes[1].plot(t, p[:, 0]+i*2, color=COLORS[i], lw=.9, label=name)
        axes[1].set(title="B  频带分解（等距错位显示）", yticks=np.arange(5)*2, yticklabels=BAND_NAMES)
        for i, (p, name) in enumerate(zip(wp, ["A3", "D3", "D2", "D1"])):
            axes[2].plot(t, p[:, 0]+i*2, color=COLORS[i], lw=.9)
        axes[2].set(title="C  db4小波重构（等距错位显示）", yticks=np.arange(4)*2, yticklabels=["A3", "D3", "D2", "D1"])
        freqs = np.geomspace(.01, .2, 56)
        wavelet = "cmor1.5-1.0"
        scales = pywt.central_frequency(wavelet) / (freqs*tr)
        coef, ff = pywt.cwt(y[:, 0]-y.mean(), scales, wavelet, sampling_period=tr)
        mesh = axes[3].pcolormesh(t, ff, np.abs(coef)**2, cmap="magma", shading="auto", rasterized=True)
        axes[3].set(yscale="log", ylabel="频率（Hz）", xlabel="时间（秒）", title="D  Morlet时频能量（边界区不用于判断瞬时事件）")
        # ±sqrt(2)*scale*TR 为本图的保守边界提醒；并非统计显著性边界。
        for side in [np.sqrt(2)*scales*tr, t[-1]-np.sqrt(2)*scales*tr]:
            axes[3].plot(np.clip(side, 0, t[-1]), ff, "w--", lw=.8)
        fig.colorbar(mesh, ax=axes[3], label="未归一化局部小波功率")
        fig.suptitle(f"{group} {version}代表序列｜{sid}｜ROI {roi+1}（频带占比接近组中位数）", fontsize=12)
        fig.savefig(output/"图片"/f"{group}_{version}代表序列分解.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tr", type=float, default=2.)
    parser.add_argument("--boundary-per-site", type=int, default=4)
    args = parser.parse_args()
    if args.tr != 2.:
        raise ValueError("本分析频带与Nyquist上限按用户确认的TR=2秒设计")
    if args.output.exists() and any((args.output/"表格").glob("*.csv")):
        raise FileExistsError("输出目录已有分析表；请另选目录以保留已有结果")
    for name in ["图片", "表格", "说明"]:
        (args.output/name).mkdir(parents=True, exist_ok=True)
    components, spectral_rows, quality_rows, boundaries, stationarity = [], [], [], [], []
    fingerprints = {}
    candidate_examples = {}
    max_errors = {"趋势": 0., "FFT": 0., "小波": 0.}
    for group, mode in [("HC", "pretrain"), ("MDD", "finetune")]:
        split_path = args.data_root/f"subject_split_{mode}.json"
        fingerprints[split_path.name] = hashlib.sha256(split_path.read_bytes()).hexdigest()
        split = json.loads(split_path.read_text())
        ids = sorted(split["train"])
        assert len(set(ids)) == len(ids)
        assert not (set(ids) & (set(split["val"]) | set(split["test"])))
        meta = pd.read_excel(args.data_root/f"Rest-meta-MDD-V1V2-Merged-{group}.xlsx")
        versions = dict(zip(meta.ID.astype(str).str.strip(), meta.Dataset.astype(str)))
        # 训练保留 merged 缺行者；只用已知ID确认的站点前缀映射补充分层标签。
        prefix_versions = {}
        for known_id, version in versions.items():
            prefix = known_id.split("-")[0]
            prefix_versions.setdefault(prefix, set()).add(version)
        missing_meta = set(ids)-versions.keys()
        # S4整站可能无merged记录；先核对全表命名规则，再用于这些缺行者。
        assert all(v == ("V2" if s.startswith("IS") else "V1") for s, v in versions.items())
        for sid in missing_meta:
            matches = prefix_versions.get(sid.split("-")[0],
                                          {"V2" if sid.startswith("IS") else "V1"})
            if len(matches) != 1:
                raise ValueError(f"无法唯一确认版本: {sid}: {matches}")
            versions[sid] = next(iter(matches))
        sites = sorted({sid.split("-")[0] for sid in ids})
        boundary_ids = set()
        for site in sites:
            site_ids = [s for s in ids if s.split("-")[0] == site]
            site_ids.sort(key=lambda s: hashlib.sha256(f"2024:{s}".encode()).hexdigest())
            boundary_ids.update(site_ids[:args.boundary_per_site])
        example_ids = set()
        for version in ["V1", "V2"]:
            ranked = sorted([s for s in ids if versions[s] == version],
                            key=lambda s: hashlib.sha256(f"example2024:{s}".encode()).hexdigest())
            example_ids.update(ranked[:32])
        for idx, sid in enumerate(ids):
            path = args.data_root/"Raw"/group/f"ROISignals_{sid}.mat"
            raw = read_mat_array(path, preferred_keys=["ROISignals", "Data"])
            if raw.ndim != 2 or raw.shape[1] < 116 or len(raw) < 96:
                raise ValueError(f"训练BOLD形状不符合K64+H32: {path}: {raw.shape}")
            raw = raw[:, :116]
            valid = np.isfinite(raw).all(0) & (np.std(raw, axis=0) > 1e-10)
            base = {"队列": group, "版本": versions[sid], "站点前缀": sid.split("-")[0], "被试": sid}
            quality_rows.append({**base, "TR数": len(raw), "ROI数": raw.shape[1],
                                 "版本来源": "站点前缀映射" if sid in missing_meta else "merged表",
                                 "非有限数值数": int((~np.isfinite(raw)).sum()),
                                 "有效ROI数": int(valid.sum()), "校准低方差ROI数": int((np.nanstd(raw[:32], 0)<1e-6).sum())})
            if not valid.any():
                continue
            x_cal = calibrate_raw(raw)[..., valid].astype(np.float64)
            # 仅形状分析去整段均值；保留校准尺度，不使用整段方差归一化作为模型输入。
            x = x_cal - x_cal.mean(0)
            variance = x.var(0)
            record_parts(components, base, "原序列", ["原序列"], x, [x])
            # 最小线性漂移与不同平滑长度；残差严格相减而得。
            tt = np.arange(len(x), dtype=float); tt -= tt.mean()
            linear = tt[:, None] * ((tt[:, None]*x).sum(0)/(tt*tt).sum())
            record_parts(components, base, "线性趋势", ["趋势", "残差"], x, [linear, x-linear])
            for window in [9, 17, 33]:
                trend = uniform_filter1d(x, window, axis=0, mode="reflect")
                err = record_parts(components, base, f"均值趋势{window}TR", ["趋势", "残差"], x, [trend, x-trend])
                max_errors["趋势"] = max(max_errors["趋势"], err)
            parts = fft_parts(x, args.tr)
            err = record_parts(components, base, "FFT", BAND_NAMES, x, parts)
            max_errors["FFT"] = max(max_errors["FFT"], err)
            for wavelet in ["db2", "db4"]:
                wp, coeffs = wave_parts(x, wavelet)
                err = record_parts(components, base, f"小波{wavelet}", ["A3", "D3", "D2", "D1"], x, wp)
                max_errors["小波"] = max(max_errors["小波"], err)
            freq, psd = signal.welch(x, fs=1/args.tr, nperseg=128, noverlap=64, axis=0, detrend="constant")
            normalized = psd/trapezoid(psd, freq, axis=0)[None, :]
            spectral_rows.append({**base, **{f"频点_{f:.9f}": float(v) for f, v in zip(freq, normalized.mean(1))}})
            ratios = np.array([p.var(0)/variance for p in parts])
            first, second = np.array_split(x, 2)
            # 比较半段频带分配，而非短窗方差导致的表面波动。
            half_ratios = []
            for half in [first, second]:
                hc = half-half.mean(0)
                half_ratios.append(np.array([p.var(0)/hc.var(0) for p in fft_parts(hc, args.tr)]).mean(1))
            stationarity.append({**base, "半段频带总变差距离": float(np.abs(half_ratios[0]-half_ratios[1]).sum()/2),
                                  "后半前半方差比中位数": float(np.median(second.var(0)/first.var(0))),
                                  "后校准方差比中位数": float(np.median(x_cal[32:].var(0))),
                                  "校准后均值绝对值中位数": float(np.median(np.abs(x_cal[32:].mean(0))))})
            if sid in boundary_ids:
                boundaries.extend(boundary_metrics(x, base, args.tr))
            key = (group, base["版本"])
            # 暂存少量确定性候选，最后按组中位数选择，避免挑最好看的ROI。
            if sid in example_ids:
                candidate_examples.setdefault(key, []).append((x, sid, ratios, np.flatnonzero(valid)))
            if (idx+1) % 200 == 0:
                print(f"{group} {idx+1}/{len(ids)}", flush=True)
    comp = pd.DataFrame(components)
    spectra = pd.DataFrame(spectral_rows)
    boundary = pd.DataFrame(boundaries)
    quality = pd.DataFrame(quality_rows)
    stable = pd.DataFrame(stationarity)
    datasets = {"逐被试数据质量": quality, "逐被试成分与自相关": comp,
                "逐被试归一化频谱": spectra, "历史可见性边界诊断": boundary,
                "逐被试半段稳定性": stable}
    for dims, name in [(["队列", "版本", "方法", "成分"], "分版本成分汇总"),
                       (["队列", "方法", "成分"], "分队列成分汇总"),
                       (["队列", "版本", "站点前缀", "方法", "成分"], "分站点成分汇总")]:
        datasets[name] = comp.groupby(dims)[["ROI平均方差比", "交叉协方差占比"]+[f"自相关{h}TR" for h in LAGS]].agg(["mean", "median", "std", "count"]).reset_index()
        datasets[name].columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in datasets[name].columns]
    for name, frame in datasets.items():
        frame.to_csv(args.output/"表格"/f"{name}.csv", index=False, encoding="utf-8-sig")
    examples, example_info = {}, []
    for key, candidates in candidate_examples.items():
        target = comp[(comp["队列"] == key[0]) & (comp["版本"] == key[1]) & (comp["方法"] == "FFT") & (comp["成分"] == BAND_NAMES[1])]["ROI平均方差比"].median()
        chosen = min(candidates, key=lambda ex: abs(ex[2][1].mean()-target))
        x, sid, ratios, roi_ids = chosen
        r = int(np.argmin(np.abs(ratios[1]-target)))
        examples[key] = (x, sid, r)
        example_info.append({"队列": key[0], "版本": key[1], "被试": sid, "ROI序号从1起": int(roi_ids[r]+1),
                             "候选规则": "每组固定SHA256排序取32人，频带占比最接近全组被试中位数；ROI同理"})
    figures(args.output, spectra, comp, boundary, examples, args.tr)
    record = {"TR秒": args.tr, "历史TR数": 64, "预测TR数": 32, "校准TR数": 32,
              "样本": quality.groupby(["队列", "版本"]).size().to_dict().__str__(),
              "总被试数": len(quality), "无效ROI数": int((116-quality["有效ROI数"]).sum()),
              "边界诊断被试数": int(boundary.groupby(["队列", "被试"]).ngroups),
              "split_sha256": fingerprints, "重构最大绝对误差": max_errors,
              "频带Hz": BANDS, "小波": ["db2", "db4"], "小波层数": 3,
              "代表序列": example_info, "聚合": "先ROI平均、再被试等权；版本分层；箱线展示被试分布，非置信区间",
              "输入说明": "仅固定manifest训练被试，复用calibrate_raw；data/Raw不等同于未预处理fMRI",
              "非因果说明": "中心均值/FFT/对称小波均为描述性整段分解；不宣称预测提升",
              "numpy": np.__version__, "scipy": __import__("scipy").__version__, "pywt": pywt.__version__}
    (args.output/"说明"/"分析记录.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
