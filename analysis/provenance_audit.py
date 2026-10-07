# coding=utf-8
"""S0：RECC 运行版本与权重 provenance 台账（RECC 执行计划 §4）。

对每个 MoE/RECC checkpoint 记录：
- SHA-256 / 大小 / arch_version / config.args 快照（RECC 相关键 + 全量 JSON 哈希）；
- 按 config.args 重建模型后的 strict load 结果（missing/unexpected 键）；
- 固定种子小型输入的前向 smoke（输出形状/有限性）；
- best/last epoch（checkpoint 内 epoch 字段缺失时标注 not_recorded，不猜测）；
- 训练 competence 口径与复评 candidate 口径分列（执行计划 §4.1 第 5 条）。

另记录仓库级证据：git HEAD、dirty 状态摘要、split manifest 文件哈希、
HC 预训练权重哈希。产物写入 analysis_results/recc_repair_20261007/
（git 忽略目录，不覆盖原始证据）。

用法::

    python -m analysis.provenance_audit \
        --ckpt checkpoints/neurotwin_tfm_finetune_moe/finetuned_best.pt [更多路径...] \
        --out_dir analysis_results/recc_repair_20261007
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from experiments.evaluate_variant import build_model_from_args, load_train_args
from utils.common import set_seed

RECC_ARGS_KEYS = [
    'name', 'mode', 'seed', 'tfm_patho_mode', 'tfm_recc_mode',
    'tfm_recc_weight', 'tfm_pair_recc_weight', 'tfm_competence_temperature',
    'tfm_competence_pcc_weight', 'tfm_recc_start_epoch',
    'tfm_recc_ramp_epochs', 'tfm_num_experts', 'tfm_top_k',
    'tfm_expert_hidden_dim', 'tfm_router_hidden_dim',
    'tfm_router_temperature', 'tfm_moe_dense_warmup_epochs',
    'tfm_moe_diversity_weight', 'tfm_router_balance_weight',
    'tfm_router_z_weight', 'tfm_router_entropy_weight',
    'tfm_patch_len', 'tfm_dim', 'tfm_layers', 'tfm_heads',
    'freeze_backbone_epochs', 'train_epochs', 'use_ema', 'amp', 'tf32',
    'data_root', 'clinical_file', 'pretrained_weight',
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _git(cmd: list) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL
                                       ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return f'<unavailable: {cmd[1]}>'


@torch.no_grad()
def audit_checkpoint(ckpt_path: Path, device: torch.device) -> dict:
    """单份权重的溯源审计（只读；strict load + 固定输入前向 smoke）。"""
    rec: dict = {'path': str(ckpt_path), 'sha256': sha256_file(ckpt_path),
                 'size_bytes': ckpt_path.stat().st_size}
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    rec['arch_version'] = raw.get('arch_version')
    cfg = raw.get('config', {}) if isinstance(raw, dict) else {}
    saved_args = cfg.get('args') if isinstance(cfg, dict) else None
    if not isinstance(saved_args, dict):
        raise ValueError(f'{ckpt_path} 缺少 config.args 快照，无法重建')
    rec['recc_args'] = {k: saved_args.get(k) for k in RECC_ARGS_KEYS}
    full = json.dumps(saved_args, sort_keys=True, ensure_ascii=False)
    rec['args_json_sha256'] = hashlib.sha256(full.encode('utf-8')).hexdigest()
    # epoch 溯源：本版本起 checkpoint 记录 epoch；历史运行无此字段，标注缺失
    for k in ('epoch', 'epoch_1based', 'val_metric', 'backbone_unfrozen',
              'dense_routing', 'recc_mode', 'candidate_schema'):
        if k in cfg:
            rec[k] = cfg[k]
    rec['epoch_provenance'] = ('checkpoint_meta' if 'epoch' in cfg
                               else 'not_recorded（需从训练日志恢复）')

    args = load_train_args(str(ckpt_path))
    model = build_model_from_args(args, device)
    compat = model.load_state_dict(raw['state_dict'], strict=True)
    rec['strict_load'] = 'ok'
    rec['n_params'] = int(sum(p.numel() for p in model.parameters()))
    rec['n_state_entries'] = len(raw['state_dict'])

    # 固定种子小型前向 smoke（形状/有限性；非历史样本复算）
    set_seed(2024)
    model.eval()
    f = int(args.num_rois)
    k = min(int(args.context_max), 32)
    x = torch.randn(2, f, 1, k, device=device)
    sc = torch.rand(f, f, device=device).unsqueeze(0).expand(2, f, f)
    score = (torch.randn(2, 1, device=device) * 5 + 18
             if args.mode == 'finetune' else None)
    is_moe = str(getattr(args, 'tfm_patho_mode', 'none')) == 'moe'
    pred, aux = model(x, sc, score, pathology_epoch=None,
                      return_expert_pred=is_moe, return_pair_pred=is_moe)
    rec['forward_smoke'] = {
        'input': [2, f, 1, k],
        'one_pred_shape': list(pred.shape),
        'finite': bool(torch.isfinite(pred).all()),
        'candidate_schema': aux.get('tfm_candidate_schema', 'not_applicable'),
    }
    # 训练 competence 口径与复评 candidate 口径分列（执行计划 §4.1-5）
    rec['training_competence_schema'] = {
        'recc_mode': saved_args.get('tfm_recc_mode'),
        'competence_temperature': saved_args.get('tfm_competence_temperature'),
        'huber_delta': 1.0,
        'note': '历史运行（≤2026-10-07）的 candidate 构造为 unanchored_v1'
                '（漏 x_last 锚点，问题核查 §3.1）；本字段仅为训练侧口径声明，'
                '复评侧口径见 forward_smoke.candidate_schema。',
    }
    return rec


def main():
    ap = argparse.ArgumentParser(description='S0 provenance 台账（只读）')
    ap.add_argument('--ckpt', nargs='+', required=True)
    ap.add_argument('--out_dir', default='analysis_results/recc_repair_20261007')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--hc_pretrained', default='',
                    help='HC 预训练权重路径（记录其 SHA-256）')
    args_cli = ap.parse_args()

    device = torch.device(args_cli.device)
    repo = Path(__file__).resolve().parents[1]
    ledger = {
        'git': {
            'head': _git(['git', '-C', str(repo), 'rev-parse', 'HEAD']),
            'branch': _git(['git', '-C', str(repo), 'rev-parse', '--abbrev-ref', 'HEAD']),
            'dirty_files': _git(['git', '-C', str(repo), 'status', '--porcelain']),
            'diff_sha256': hashlib.sha256(
                _git(['git', '-C', str(repo), 'diff']).encode()).hexdigest(),
            'note': 'dirty_files 为生成时快照；diff_sha256 是当前未提交改动'
                    '（不含 staged/untracked）的指纹',
        },
        'split_manifests': {},
        'hc_pretrained_sha256': (sha256_file(Path(args_cli.hc_pretrained))
                                 if args_cli.hc_pretrained else None),
        'checkpoints': [],
    }
    for manifest in sorted((repo / 'data').glob('subject_split_*.json')):
        ledger['split_manifests'][manifest.name] = {
            'sha256': sha256_file(manifest),
            'subjects': len(json.loads(manifest.read_text()).get(
                'train', manifest)),
        }
    for p in args_cli.ckpt:
        path = Path(p)
        print(f'[provenance] {path} ...')
        ledger['checkpoints'].append(audit_checkpoint(path, device))

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'provenance_ledger.json'
    out.write_text(json.dumps(ledger, ensure_ascii=False, indent=2),
                   encoding='utf-8')
    rows = ['path,sha256,arch_version,strict_load,n_params,epoch_provenance,'
            'candidate_schema']
    for r in ledger['checkpoints']:
        rows.append(','.join([
            r['path'], r['sha256'][:16], str(r.get('arch_version')),
            r.get('strict_load', 'FAIL'), str(r.get('n_params')),
            r['epoch_provenance'],
            str(r.get('forward_smoke', {}).get('candidate_schema'))]))
    (out_dir / 'provenance_summary.csv').write_text('\n'.join(rows) + '\n',
                                                    encoding='utf-8')
    print(f'[provenance] 台账写入 {out}')
    for r in ledger['checkpoints']:
        ok = r.get('strict_load') == 'ok' and r['forward_smoke']['finite']
        print(f"  {'OK ' if ok else 'FAIL'} {Path(r['path']).name} "
              f"arch_v={r.get('arch_version')} params={r.get('n_params')} "
              f"epoch={r['epoch_provenance']}")
    if any(r.get('strict_load') != 'ok'
           or not r['forward_smoke']['finite']
           for r in ledger['checkpoints']):
        sys.exit(1)


if __name__ == '__main__':
    main()
