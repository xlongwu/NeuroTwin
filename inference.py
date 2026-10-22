# coding=utf-8
"""NeuroTwinTFM 最终定型主模型：加载与推理入口。

最终主模型 = **NeuroTwinTFM + SC 软先验 + 单路 FiLM + One-Step/CPM 双头（patch=4）**，
定型决定与会话记录见 TASK_HANDOFF.md（2026-10-07 第四次会话）；
最终配置的唯一事实来源为 configs/final_model.json（含结构签名、参数量与权重
sha256），本模块消费该文件。

固化范围（2026-10-07 定型，零训练）：
  - 主权重 strict 加载（结构不匹配明确报错，不使用 strict=False 绕过）；
  - checkpoint 内保存的 HAMD normalizer buffers 随权重加载，禁止在 val/test
    拟合或替换归一化统计量；
  - 推理接口复用模型既有方法：下一状态 ``predict_next_state``（One-Step：
    residual → 加 x_last_norm → 同 context 统计量 denorm）；2–8 步整段预测
    ``forecast_horizon``（CPM 一次非自回归输出，不额外加 One-Step 锚点，
    不用 One-Step 替换 CPM 第一帧）；
  - 主配置请求超过 8 步时明确报错：直接 CPM 支持范围 H≤8；H16 长程参考是
    B（MoE）权重的固定 CPM 分块滚动，属独立归档配置，不并入本入口，
    不静默切换权重或 rollout 协议。

输入口径（沿用训练版本，不在入口更换滤波或标准化协议）：
  - ``bold_history``：上游逐 ROI 全序列 z-score 后的 BOLD context [B, K, F]，
    K ≤ context_max=64（训练分布 K∈[16,64]，评估协议固定 K=64）；
  - ``sc_matrix``：数据侧 ``NeuroTwinNextPointDataset._normalize_sc`` 清洗后
    的 SC（nan→0、对称化、clip(0)、log1p、99 分位归一、clip[0,1]）；原始
    Mask 矩阵须先经 :func:`normalize_raw_sc` 处理；
  - ``pathology_score``：原始 HAMD 评分 [B] 或 [B,1]，最终主模型为 FiLM 条件
    模型，缺省即报错（无条件的无条件推理请直接使用 models.tfm.NeuroTwinTFM）。

用法::

    from inference import load_final_model, predict_next_state, forecast_horizon
    model = load_final_model()                       # 默认主权重 + 哈希校验
    out = predict_next_state(model, hist, sc, hamd)  # out['pred_next'] [B,F,1]
    traj = forecast_horizon(model, hist, sc, hamd, steps=8)  # [B,8,F]

    python inference.py                              # CLI 自检：加载 + 校验 + 摘要
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from models import ARCH_VERSION
from experiments.evaluate_variant import build_model_from_args, load_train_args

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / 'configs' / 'final_model.json'


def load_final_config(path=None) -> dict:
    """读取最终定型配置（configs/final_model.json）。"""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _default_weight_path(cfg) -> Path:
    """定型主权重路径（**不**检查存在性）。

    显式传入 ``ckpt_path`` 时也必须先知道默认路径才能判断“是否在用定型主权重”，
    但判断不该要求该文件存在：早期实现在这里直接 ``FileNotFoundError``，
    导致“我只想看另一个权重”也被默认路径拦住（3 个闸门测试因此全部撞在环境问题上）。
    """
    p = Path(cfg['weights']['main']['path'])
    return p if p.is_absolute() else ROOT / p


def _require_weight(path: Path) -> Path:
    """确实要加载该路径时才要求存在（错误信息保留权重不入版本库的提示）。"""
    if not path.exists():
        raise FileNotFoundError(
            f"最终主权重不存在: {path}（configs/final_model.json 记录的路径；"
            "权重不入版本库，请确认 checkpoints/ 已就位）")
    return path


def _expected_arch_version(cfg) -> int:
    """定型入口的**隔离版本闸门**：只认配置里结构签名声明的 arch_version。

    条件注入实验把 ``models.ARCH_VERSION`` 抬到了 5（新增条件模块），但定型主权重是
    2026-10-07 冻结的 v4 工件。本入口按配置声明的版本校验，因此：
      - 不靠“改版本号”把旧权重说成已迁移；
      - v5 的条件实验权重也不允许冒充定型主权重（会在这里被拒绝，请走
        ``experiments.evaluate_variant``）；
      - 结构等价性由紧随其后的 strict 加载与参数量校验保证，而不是由版本号保证。
    """
    sig = (cfg.get('structure_signature') or {})
    if 'arch_version' not in sig:
        raise RuntimeError(
            'configs/final_model.json 的 structure_signature 缺少 arch_version：'
            '定型入口无法确定该工件的结构版本，拒绝加载（不猜测当前代码版本）。')
    return int(sig['arch_version'])


def load_final_model(ckpt_path=None, device=None, verify_hash=True,
                     config_path=None):
    """按最终配置加载主模型（FiLM 闸门 + strict 加载 + 完整性校验）。

    Args:
        ckpt_path: 权重路径；None = configs/final_model.json 记录的主权重。
            传入其他 FiLM checkpoint 时跳过哈希/参数量校验，但 FiLM 闸门、
            strict 加载与 normalizer 校验仍然生效（MoE/RECC/HC 权重会被拒绝）。
        device: None = cuda:0（可用时）否则 cpu。
        verify_hash: 对默认主权重做 sha256 完整性校验（与配置记录比对）。
        config_path: 替代配置文件路径（默认 configs/final_model.json）。

    Returns:
        eval 模式的 NeuroTwinTFM（HAMD normalizer buffers 已随权重加载）。
    """
    cfg = load_final_config(config_path)
    default_path = _default_weight_path(cfg)
    if ckpt_path is None:
        ckpt_path = _require_weight(default_path)
        is_default = True
    else:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_absolute():
            ckpt_path = (ROOT / ckpt_path) if not ckpt_path.exists() else ckpt_path
        is_default = (ckpt_path.exists()
                      and ckpt_path.resolve() == default_path.resolve())

    if is_default and verify_hash:
        actual = _sha256(ckpt_path)
        expected = cfg['weights']['main']['sha256']
        if actual != expected:
            raise RuntimeError(
                f"主权重 sha256 与最终配置记录不一致：\n  实际 {actual}\n  记录 {expected}"
                f"\n该文件不是定型时校验过的权重（{ckpt_path}）；如确认要加载其它"
                "文件，请显式传入 --ckpt 或 verify_hash=False，但不得宣称其为定型主权重。")

    device = torch.device(device) if device is not None else torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')

    # 架构版本闸门：结构改动会改变参数语义，跨版本权重不允许静默加载
    expected_arch = _expected_arch_version(cfg)
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    arch_version = raw.get('arch_version') if isinstance(raw, dict) else None
    if arch_version is None:
        raise RuntimeError(
            f"{ckpt_path} 缺少 arch_version 元数据（非当前 save_backbone_weights "
            f"格式），无法确认结构版本；最终入口拒绝加载。")
    if int(arch_version) != expected_arch:
        raise RuntimeError(
            f"权重 arch_version={arch_version} 与定型入口要求的 "
            f"arch_version={expected_arch} 不一致（{ckpt_path}）：结构版本不同的权重不得混用。"
            f"（当前代码 ARCH_VERSION={ARCH_VERSION}；条件注入等结构变更属另一条实验线，"
            "其权重请用 experiments.evaluate_variant 加载，不并入本定型入口。）")

    # FiLM 闸门：最终主路径不使用 MoE/Router/RECC。MoE checkpoint 的快照会把
    # 模型重建为 tfm_patho_mode='moe'，HC 权重重建为 'none'，均在此明确拒绝。
    model = build_model_from_args(load_train_args(ckpt_path), device)
    if getattr(model, 'tfm_patho_mode', None) != 'film':
        raise RuntimeError(
            f"该 checkpoint 重建为 tfm_patho_mode="
            f"'{getattr(model, 'tfm_patho_mode', None)}'，不是最终 FiLM 主权重"
            f"（{ckpt_path}）。MoE/RECC 变体请按其归档方式单独加载（"
            "experiments.evaluate_variant），不得混入最终主路径。")

    # strict 加载：结构不匹配直接抛错，不使用 strict=False 绕过
    from analysis.checkpoint import load_checkpoint_compat
    try:
        model, compat = load_checkpoint_compat(model, ckpt_path, device, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"主权重 strict 加载失败（结构不匹配，{ckpt_path}）：{e}\n"
            "不使用 strict=False 绕过；请确认权重与 configs/final_model.json 的"
            "结构签名一致。") from e
    if compat.get('missing') or compat.get('unexpected'):
        raise RuntimeError(f"权重加载不完整（不允许静默缺口）：{compat}")

    # HAMD normalizer buffers 必须来自 checkpoint（fitted 标志在位）；
    # 本入口不做任何 fit/替换统计量的操作
    norm = getattr(model, 'pathology_normalizer', None)
    if norm is None or not norm.is_fitted():
        raise RuntimeError(
            f"{ckpt_path} 未携带已拟合的 HAMD normalizer buffers：最终主模型的"
            "条件统计量必须来自 checkpoint，禁止在 val/test 重新拟合。")

    # 默认主权重：核对冻结参数量与定型记录一致（防同名不同量）
    if is_default:
        n_total = sum(p.numel() for p in model.parameters())
        n_film = sum(p.numel() for m in model.modules()
                     if getattr(m, 'is_conditioning_adapter', False)
                     for p in m.parameters())
        ref = cfg['param_counts']
        if n_total != ref['total'] or n_film != ref['film_total']:
            raise RuntimeError(
                f"参数量与定型记录不一致：实际 total={n_total}, FiLM={n_film}；"
                f"记录 total={ref['total']}, FiLM={ref['film_total']}。"
                "该权重与 configs/final_model.json 不匹配。")

    model.eval()
    return model


def normalize_raw_sc(sc_matrix):
    """原始结构连接矩阵 → 模型输入口径（复用数据侧同一清洗实现）。"""
    from utils.dataloader import NeuroTwinNextPointDataset
    return NeuroTwinNextPointDataset._normalize_sc(sc_matrix)


def _validate_inputs(model, bold_history, sc_matrix, pathology_score):
    """推理入口的输入契约校验（形状/维度清晰报错；语义校验仍在模型内）。"""
    if not torch.is_tensor(bold_history):
        raise TypeError(f"bold_history 须为 torch.Tensor，收到 {type(bold_history)}")
    if bold_history.ndim != 3:
        raise ValueError(
            f"bold_history 期望 [B, K, F]，收到 {tuple(bold_history.shape)}")
    b, k, f = bold_history.shape
    if f != model.features:
        raise ValueError(
            f"bold_history 最后一维须为 ROI 数 {model.features}，收到 {f}")
    if not (1 <= k <= model.context_max):
        raise ValueError(
            f"context 长度 K 须在 [1, context_max={model.context_max}] 内，收到 {k}"
            "（训练分布 K∈[16,64]，评估协议固定 K=64，见 configs/final_model.json）")
    if not torch.is_tensor(sc_matrix):
        raise TypeError(f"sc_matrix 须为 torch.Tensor，收到 {type(sc_matrix)}")
    if sc_matrix.ndim not in (2, 3):
        raise ValueError(
            f"sc_matrix 期望 [F, F] 或 [B, F, F]，收到 {tuple(sc_matrix.shape)}")
    if sc_matrix.shape[-1] != f or sc_matrix.shape[-2] != f:
        raise ValueError(
            f"sc_matrix 尺寸须为 {f}×{f}，收到 {tuple(sc_matrix.shape[-2:])}；"
            "原始 Mask 矩阵请先经 normalize_raw_sc 清洗。")
    if pathology_score is None:
        raise ValueError(
            "最终主模型为 HAMD FiLM 条件模型：pathology_score 为必要输入"
            "（[B] 或 [B,1] 的原始 HAMD 评分，模型内按 checkpoint 的 robust_z "
            "统计量归一化）；无条件推理请直接使用 models.tfm.NeuroTwinTFM。")
    if not torch.is_tensor(pathology_score):
        raise TypeError(f"pathology_score 须为 torch.Tensor，收到 {type(pathology_score)}")
    if pathology_score.ndim not in (1, 2) or pathology_score.shape[0] != b:
        raise ValueError(
            f"pathology_score 期望 [B] 或 [B, D] 且 B={b}，"
            f"收到 {tuple(pathology_score.shape)}")


def predict_next_state(model, bold_history, sc_matrix, pathology_score) -> dict:
    """下一状态预测（One-Step 头；复用 NeuroTwinTFM.predict_next_state）。

    返回 dict：``pred_next`` [B,F,1]（原始空间）、``pred_delta`` [B,F,1]、
    ``pred`` [B,F,1,1]、``aux_info``（含 CPM 的 ``cpm_pred`` [B,F,H]）。
    One-Step 链路：residual → 加 x_last_norm → 同 context 统计量 denorm。
    """
    _validate_inputs(model, bold_history, sc_matrix, pathology_score)
    return model.predict_next_state(bold_history, sc_matrix, pathology_score)


def forecast_horizon(model, bold_history, sc_matrix, pathology_score,
                     steps=None) -> torch.Tensor:
    """2–8 步整段预测（CPM 头；复用 NeuroTwinTFM.forecast_horizon）。

    CPM 一次非自回归输出 t+1..t+H 完整状态并同 context 统计量反归一化；
    不额外添加 One-Step 锚点，也不用 One-Step 替换 CPM 第一帧。

    Args:
        steps: 请求的步数（1..cpm_horizon=8）；None = 全部 H=8。请求
            ``steps < H`` 时返回 CPM 完整轨迹的前 steps 帧（非自回归，
            与一次性输出逐位一致）；``steps > H`` 明确报错。

    Returns:
        [B, steps, F]（原始空间）。
    """
    _validate_inputs(model, bold_history, sc_matrix, pathology_score)
    horizon = int(model.cpm_horizon)
    if steps is None:
        steps = horizon
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"steps 必须 >= 1，收到 {steps}")
    if steps > horizon:
        raise RuntimeError(
            f"请求 {steps} 步超过主模型直接 CPM 支持范围 H≤{horizon}"
            f"（cpm_horizon={horizon}）：不静默切换到 B 权重或其它 rollout 协议。"
            "H16 长程参考是 B（MoE）best 的固定 CPM 分块滚动"
            "（checkpoints/neurotwin_tfm_finetune_moe/finetuned_best.pt，独立归档"
            "配置，未并入统一 FiLM 主模型）；如需更长轨迹请显式使用该独立参考或"
            "将 H≤8 的 CPM 结果作为分段输入重新评估。")
    traj = model.forecast_horizon(bold_history, sc_matrix, pathology_score)
    return traj[:, :steps, :]


def main():
    """CLI 自检：加载最终主权重并打印定型摘要（不做任何训练/评估）。"""
    ap = argparse.ArgumentParser(
        description='NeuroTwinTFM 最终主模型加载自检（零训练推理入口）')
    ap.add_argument('--ckpt', default=None,
                    help='权重路径；默认取 configs/final_model.json 记录的主权重')
    ap.add_argument('--config', default=None,
                    help='配置文件路径；默认 configs/final_model.json')
    ap.add_argument('--device', default=None,
                    help='推理设备（如 cpu / cuda:0）；默认 cuda:0 可用时优先')
    ap.add_argument('--no_verify_hash', action='store_true',
                    help='跳过主权重 sha256 校验（默认校验）')
    args = ap.parse_args()

    cfg = load_final_config(args.config)
    model = load_final_model(ckpt_path=args.ckpt, device=args.device,
                             verify_hash=not args.no_verify_hash,
                             config_path=args.config)
    n_total = sum(p.numel() for p in model.parameters())
    n_film = sum(p.numel() for m in model.modules()
                 if getattr(m, 'is_conditioning_adapter', False)
                 for p in m.parameters())
    sig = cfg['structure_signature']
    print('[final-model] 结构签名: ' + json.dumps(sig, ensure_ascii=False))
    print(f"[final-model] tfm_patho_mode={model.tfm_patho_mode} | "
          f"patch_len={model.patch_len} | cpm_horizon={model.cpm_horizon} | "
          f"context_max={model.context_max} | revin={model.use_revin}")
    print(f"[final-model] 参数量: total={n_total} | FiLM={n_film} | "
          f"normalizer(mode={model.pathology_normalizer.mode}, "
          f"fitted={model.pathology_normalizer.is_fitted()})")
    print(f"[final-model] 设备: {next(model.parameters()).device} | "
          "strict 加载成功，无 missing/unexpected keys")


if __name__ == '__main__':
    main()
