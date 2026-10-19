"""Generate the 21 research figures from native data and trained checkpoints.

Run from the repository root: python -m analysis.run_visualizations
No reports, result tables, synthetic data, training or checkpoint writes.
"""
import argparse
from pathlib import Path

import torch

from analysis import visualization_data as data
from analysis import visualization_plots as plots


ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=ROOT/'data')
    p.add_argument('--hc-checkpoint', type=Path, default=ROOT/'checkpoints/neurotwin_tfm_pretrain/base_best.pt')
    p.add_argument('--mdd-checkpoint', type=Path, default=ROOT/'checkpoints/neurotwin_tfm_finetune/finetuned_best.pt')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/analysis_figures')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--anchors', type=int, default=16)
    p.add_argument('--max-subjects', type=int, default=0,
                   help='Deterministic test subset per group; 0 uses every frozen test subject. HC training reference remains complete.')
    args = p.parse_args(argv)
    if min(args.batch_size, args.threads, args.anchors) < 1 or args.max_subjects < 0:
        p.error('Batch size, threads and anchors must be positive; max-subjects must be nonnegative')
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    plots.style()
    print('Loading native cohorts, frozen splits and strict checkpoint state dictionaries', flush=True)
    d = data.load_inputs(args)
    data.compute_forecasts(d)
    data.compute_dynamics(d)
    plots.forecast_figures(d)
    plots.dynamics_figures(d)
    plots.pathology_figures(d)
    data.compute_normative(d)
    plots.normative_figures(d)
    print('Analyses 1–4 complete; computing ROI occlusion and clinical analyses', flush=True)
    data.compute_explanations(d)
    plots.explanation_figures(d)
    data.compute_clinical(d)
    plots.clinical_figures(d)
    data.compute_sensitivity(d)
    plots.sensitivity_figures(d)
    plots.uncertainty_figures(d)
    print(f'Completed 21 PNG figures at 300 DPI: {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
