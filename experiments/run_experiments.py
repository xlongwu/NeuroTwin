# coding=utf-8
"""消融实验批量执行入口。

用法示例：
  python -m experiments.run_experiments --group G14_MOE --dry-run      # 只打印命令
  python -m experiments.run_experiments --only baseline --smoke        # 端到端冒烟
  python -m experiments.run_experiments --priority P0 --seeds 2024     # 正式 P0 组

流程：合并参数（BASE_ARGS + 变体 overrides + 运行参数）→ 可选自预训练 →
训练（subprocess，日志落盘）→ 轻量评估（val/test + shuffled 负对照）→
registry 登记。已 success 的 (exp_id, seed) 幂等跳过（--force 重跑）。
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import registry
from experiments.base_config import (BASE_ARGS, CHECKPOINT_DIR,
                                     build_cli_args, validate_overrides)
from experiments.variants import filter_variants

# smoke 模式覆盖：最小成本验证框架端到端，不追求效果
SMOKE_OVERRIDES = {
    'train_epochs': 2,
    'warmup_epochs': 1,
    'patience': 1,
    'batch_size': 4,
    'num_workers': 0,
    'compile': False,
    'eval_batch_size': 8,
    'cache_in_memory': False,
}

PRIORITY_ORDER = {'P0': 0, 'P1': 1, 'P2': 2}
LOGS_DIR = registry.RESULTS_DIR / 'logs'
EVAL_DIR = registry.RESULTS_DIR / 'eval'


def merge_run_args(variant, seed, smoke, pretrain_stage=False):
    """合并 BASE_ARGS + 变体 overrides + 运行参数，返回 (args_dict, name)。"""
    merged = dict(BASE_ARGS)
    merged.update(variant['overrides'])
    if smoke:
        merged.update(SMOKE_OVERRIDES)
    merged['seed'] = seed
    if pretrain_stage:
        name = f"ablation_pre_{variant['id']}_s{seed}"
        merged['mode'] = 'pretrain'
        # 预训练不消费预训练权重：去掉该键，避免歧义
        merged.pop('pretrained_weight', None)
    else:
        name = f"ablation_ft_{variant['id']}_s{seed}"
        merged['mode'] = 'finetune'
        merged['pretrained_weight'] = resolve_pretrained_weight(merged, variant)
    if smoke:
        # 冒烟使用独立目录名：main.py 在目录名冲突时会改存到带数字后缀的
        # 新目录，导致按原始名解析权重时加载到冒烟残留（已发生过的事故）
        name = f"{name}_smoke"
    merged['name'] = name
    return merged, name


def resolve_pretrained_weight(merged, variant):
    """变体微调的预训练权重来源：结构性变体用自预训练产物，其余复用 next_timepoint 基线。"""
    if variant['requires_pretrain']:
        return str(CHECKPOINT_DIR / f"ablation_pre_{variant['id']}_s{merged['seed']}"
                   / 'base_best.pt')
    return str(CHECKPOINT_DIR / 'neurotwin_nextpoint_pretrain' / 'base_best.pt')


def resolve_run_ckpt(name, filename='finetuned_best.pt'):
    """解析 main.py 实际写入的训练产物路径。

    main.py 在目录名冲突时会追加数字后缀（name -> name2），因此不能直接用
    name/filename 定位；原名与后缀目录并存时取权重 mtime 最新者（即最近
    一次训练的产物）。
    """
    candidates = [CHECKPOINT_DIR / name, *CHECKPOINT_DIR.glob(f"{name}[0-9]*")]
    valid = [d for d in candidates if (d / filename).exists()]
    if not valid:
        raise FileNotFoundError(
            f'未找到训练权重：{CHECKPOINT_DIR / name / filename}')
    return max(valid, key=lambda d: (d / filename).stat().st_mtime) / filename


def run_training(merged, log_path):
    """启动一次 main.py 训练，stdout/stderr 落入日志文件。"""
    cmd = [sys.executable, '-u', 'main.py'] + build_cli_args(merged)
    cmd_str = ' '.join(cmd)
    print(f"[CMD] {cmd_str}")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"[CMD] {cmd_str}\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(registry.ROOT), stdout=f,
                              stderr=subprocess.STDOUT)
    return proc.returncode, cmd_str


def tail_log(log_path, n_lines=30):
    path = Path(log_path)
    if not path.exists():
        return '(日志文件不存在)'
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    return '\n'.join(lines[-n_lines:])


def run_variant(variant, seed, smoke=False, dry_run=False):
    """执行单个变体：训练（+可选自预训练）→ 评估 → 登记。返回 True/False。"""
    exp_id = variant['id']
    log_path = LOGS_DIR / f"{exp_id}_s{seed}.log"
    started = registry.now_str()

    if dry_run:
        merged, _ = merge_run_args(variant, seed, smoke)
        validate_overrides(variant['overrides'])
        cmd = [sys.executable, '-u', 'main.py'] + build_cli_args(merged)
        print(f"[CMD] {' '.join(cmd)}")
        if variant['requires_pretrain']:
            merged_pre, _ = merge_run_args(variant, seed, smoke,
                                           pretrain_stage=True)
            cmd_pre = [sys.executable, '-u', 'main.py'] + build_cli_args(merged_pre)
            print(f"[CMD] {' '.join(cmd_pre)}   # 自预训练（先行）")
        return True

    record = dict(
        exp_id=exp_id, group=variant['group'], doc_ref=variant['doc_ref'],
        priority=variant['priority'], seed=seed, mode='finetune',
        status='running', git_commit=registry.get_git_commit(),
        started_at=started, cmd=None, log_path=str(log_path),
        note=variant['description'],
    )

    try:
        # 前置检查：复用的基线预训练权重必须存在（结构性变体则自产）
        merged_ft, ft_name = merge_run_args(variant, seed, smoke)
        validate_overrides(variant['overrides'])
        if not variant['requires_pretrain'] and \
                not os.path.exists(merged_ft['pretrained_weight']):
            raise FileNotFoundError(
                f"未找到基线预训练权重：{merged_ft['pretrained_weight']}")

        if variant['requires_pretrain']:
            merged_pre, pre_name = merge_run_args(variant, seed, smoke,
                                                  pretrain_stage=True)
            print(f"=== [{exp_id} s{seed}] 自预训练（结构变体）: {pre_name} ===")
            code, cmd_str = run_training(
                merged_pre, LOGS_DIR / f"{exp_id}_s{seed}_pre.log")
            if code != 0:
                raise RuntimeError(f'自预训练失败（exit={code}）')
            # 目录冲突时 main.py 会改存到带后缀的目录，按实际产物解析
            merged_ft['pretrained_weight'] = str(
                resolve_run_ckpt(pre_name, 'base_best.pt'))

        print(f"=== [{exp_id} s{seed}] 微调: {ft_name} ===")
        code, cmd_str = run_training(merged_ft, log_path)
        if code != 0:
            raise RuntimeError(f'训练失败（exit={code}）')

        ckpt = resolve_run_ckpt(ft_name)
        print(f"=== [{exp_id} s{seed}] 评估 ===")
        summary = evaluate_ckpt(ckpt, EVAL_DIR / f"{exp_id}_s{seed}", seed=seed)

        # smoke 记录不入正式 success 状态，避免 is_done 跳过后续正式运行
        record.update(
            status='smoke_success' if smoke else 'success',
            cmd=cmd_str,
            finished_at=registry.now_str(),
            val_pcc=summary.get('val_pcc'), val_mae=summary.get('val_mae'),
            test_pcc=summary.get('test_pcc'), test_mae=summary.get('test_mae'),
            test_r2=summary.get('test_r2'), test_picp=summary.get('test_picp'),
            test_picp_cal=summary.get('test_picp_cal'),
            test_fc_upper_mae=summary.get('test_fc_upper_mae'),
            test_edge_pcc=summary.get('test_edge_pcc'),
            test_subj_pcc=summary.get('test_subj_pcc'),
            test_n_params=summary.get('test_n_params'),
        )
        registry.upsert_record(record)
        print(f"[OK] {exp_id} s{seed}: test_pcc={record['test_pcc']}, "
              f"test_mae={record['test_mae']}")
        return True

    except Exception as exc:  # noqa: BLE001 —— 单变体失败不阻断批量
        record.update(
            status='failed', cmd=record.get('cmd'),
            finished_at=registry.now_str(),
            note=f"{variant['description']} | 失败: {exc}",
        )
        registry.upsert_record(record)
        print(f"[FAIL] {exp_id} s{seed}: {exc}")
        print(f"----- 日志尾部（{log_path}）-----")
        print(tail_log(log_path))
        return False


def evaluate_ckpt(ckpt, out_dir, seed):
    """隔离导入：仅训练成功后才加载 torch 评估栈。"""
    from experiments.evaluate_variant import evaluate
    return evaluate(str(ckpt), str(out_dir), eval_splits=('test',),
                    seed=seed, run_shuffled=True)


def main():
    ap = argparse.ArgumentParser(description='NeuroTwin 消融实验批量执行')
    ap.add_argument('--group', default=None, help='实验组（如 G14_MOE）')
    ap.add_argument('--only', default=None, help='逗号分隔的实验 ID 列表')
    ap.add_argument('--priority', default=None, choices=['P0', 'P1', 'P2'])
    ap.add_argument('--seeds', default='2024', help='逗号分隔的种子列表')
    ap.add_argument('--smoke', action='store_true', help='冒烟模式（2 epochs 小 batch）')
    ap.add_argument('--dry-run', action='store_true', help='只打印命令不执行')
    ap.add_argument('--force', action='store_true', help='重跑已 success 的变体')
    ap.add_argument('--stop-on-fail', action='store_true', help='首个失败即中止')
    ns = ap.parse_args()

    seeds = [int(s) for s in ns.seeds.split(',') if s.strip()]
    if ns.only:
        variants = filter_variants(only=ns.only)
    else:
        variants = filter_variants(group=ns.group, priority=ns.priority)
    # P0 先跑，同优先级保持注册顺序
    variants = sorted(variants, key=lambda v: PRIORITY_ORDER.get(v['priority'], 9))
    if not variants:
        print('没有匹配的变体，请检查 --group/--only/--priority。')
        return 0

    print(f'待执行 {len(variants)} 个变体 x {len(seeds)} 个 seed '
          f'（smoke={ns.smoke}, dry_run={ns.dry_run}）')
    failed = []
    for variant in variants:
        for seed in seeds:
            if not ns.force and not ns.dry_run and \
                    registry.is_done(variant['id'], seed):
                print(f"[SKIP] {variant['id']} s{seed} 已 success（--force 重跑）")
                continue
            ok = run_variant(variant, seed, smoke=ns.smoke, dry_run=ns.dry_run)
            if not ok:
                failed.append((variant['id'], seed))
                if ns.stop_on_fail:
                    print('--stop-on-fail 已触发，中止后续实验。')
                    return 1
    if failed:
        print(f'完成，但 {len(failed)} 个实验失败：{failed}')
        return 1
    print('全部完成。对比报告：python -m experiments.compare'
          + (f' --group {ns.group}' if ns.group else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
