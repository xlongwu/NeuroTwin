"""Reproducible unified study CLI. Config paths are relative to the current repository."""

import argparse
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from analysis import study_stages as stages
from analysis.study_adapter import NativeAdapter, SyntheticAdapter
from analysis.study_data import native_subjects, sha256, synthetic_subjects

STAGES = (
    "audit",
    "performance",
    "dynamics",
    "pathology",
    "normative",
    "clinical",
    "explainability",
    "sensitivity",
    "uncertainty",
    "report",
)
DEFAULTS = dict(
    seed=2024,
    source="native",
    data_root="data",
    checkpoint="checkpoints/neurotwin_tfm_pretrain/base_best.pt",
    output_root="results/analysis",
    run_name="study",
    device="cpu",
    torch_threads=2,
    cohorts=["HC", "MDD"],
    max_subjects=dict(train=24, val=6, test=6),
    evaluation_split="test",
    contexts=[64],
    horizons=[1, 2, 4, 8],
    prediction_modes=["one_step", "direct_cpm", "ode_integration", "autoregressive"],
    anchors_per_subject=2,
    bootstrap_repeats=1000,
    dynamics_horizon=64,
    min_dynamics_length=64,
    dfc_window=32,
    dfc_stride=8,
    tr=None,
    min_reference_subjects=12,
    covariates=["Age", "Sex", "site", "FD", "medication"],
    network_mapping=None,
    explain_subjects_per_cohort=2,
    explain_max_rois=8,
    explain_mode="ode_integration",
    ig_steps=16,
    parameter_randomization=True,
)


def load_config(path):
    if path is None:
        payload = {}
    else:
        text = Path(path).read_text(encoding="utf-8")
        if Path(path).suffix in (".yaml", ".yml"):
            import yaml

            payload = yaml.safe_load(text)
        else:
            payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Configuration must be an object")
    unknown = set(payload) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    config = {**DEFAULTS, **payload}
    config["max_subjects"] = {
        **DEFAULTS["max_subjects"],
        **payload.get("max_subjects", {}),
    }
    if config["source"] not in ("native", "synthetic") or config[
        "evaluation_split"
    ] not in ("val", "test"):
        raise ValueError("Invalid source or evaluation split")
    if (
        not config["contexts"]
        or not config["horizons"]
        or min(config["contexts"] + config["horizons"]) < 1
    ):
        raise ValueError("Contexts/horizons must be nonempty positive integers")
    if (
        config["bootstrap_repeats"] < 10
        or config["anchors_per_subject"] < 1
        or config["min_dynamics_length"] < 32
    ):
        raise ValueError("Insufficient bootstrap/anchor count or dynamics minimum")
    if not set(config["cohorts"]).issubset({"HC", "MDD"}) or not config["cohorts"]:
        raise ValueError("cohorts must be HC and/or MDD")
    if set(config["prediction_modes"]) - {
        "one_step",
        "direct_cpm",
        "ode_integration",
        "autoregressive",
    }:
        raise ValueError("Invalid prediction mode")
    if any(v < 0 for v in config["max_subjects"].values()):
        raise ValueError("Negative subject budget")
    if Path(config["run_name"]).name != config["run_name"] or config["run_name"] in (
        ".",
        "..",
    ):
        raise ValueError("run_name must be a single directory name")
    return config


def read_git_value(args):
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, obj) -> None:
    """写入标准 JSON；未定义统计量由表格保留为空值。"""
    # JSON standard has no NaN/Infinity. Tables preserve undefined numerical values as blanks.
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def render_report(output_dir, synthetic, status):
    from analysis.study_plots import render

    figures = render(output_dir, synthetic)
    lines = [
        "# Study run report",
        "",
        f"Data: {'SYNTHETIC TEST DATA' if synthetic else 'native, bounded subset'}",
        "",
        "| Stage | Status | Reason |",
        "|---|---|---|",
    ]
    lines += [f"| {k} | {v['status']} | {v['reason']} |" for k, v in status.items()]
    lines += [
        "",
        "Figures: " + ", ".join(figures),
        "",
        "All inferential units are subjects. No longitudinal/real intervention validation. Clinical conditioning is model sensitivity only.",
        "Network summaries require a supplied ROI mapping with an explicit source. Missing site/FD are not controlled.",
        "Short forecast segments are never stitched for FC/PSD/dFC. Normative training residuals are not calibrated percentiles.",
        "Local scripts do not establish that historical checkpoint selection was independent of test results.",
        "See config.json, provenance.json, status.json, run.log and per-subject CSVs for the exact analysis population.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return figures


def run(config, stage="all", output=None):
    output_dir = (
        Path(output)
        if output
        else Path(config["output_root"])
        / (
            config["run_name"]
            + "_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    status = {}
    write_json(output_dir / "config.json", config)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["torch_threads"])
    provenance = dict(
        git_commit=read_git_value(["rev-parse", "HEAD"]),
        worktree=read_git_value(["status", "--short"]),
        python=platform.python_version(),
        numpy=np.__version__,
        torch=torch.__version__,
        argv=sys.argv,
        dependencies={
            name: version(name)
            for name in ("pandas", "scipy", "matplotlib", "openpyxl")
        },
        source_sha256={
            str(path): sha256(path)
            for path in sorted(
                set(Path("analysis").glob("study_*.py"))
                | {
                    Path("analysis/run_study.py"),
                    Path("analysis/twin_reference.py"),
                    Path("models/tfm.py"),
                    Path("models/tfm_dynamics.py"),
                    Path("utils/dataloader.py"),
                }
            )
        },
        synthetic=config["source"] == "synthetic",
        seed=config["seed"],
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    try:
        if config["source"] == "native" and not Path(config["checkpoint"]).is_file():
            status["inputs"] = dict(
                status="SKIPPED_MISSING_CHECKPOINT",
                reason="MISSING_DATA: checkpoint file does not exist",
            )
            write_json(output_dir / "provenance.json", provenance)
            write_json(output_dir / "status.json", status)
            render_report(output_dir, False, status)
            return output_dir, status
        if config["source"] == "native" and not Path(config["data_root"]).is_dir():
            status["inputs"] = dict(
                status="SKIPPED_MISSING_DATA",
                reason="MISSING_DATA: data root does not exist",
            )
            write_json(output_dir / "provenance.json", provenance)
            write_json(output_dir / "status.json", status)
            render_report(output_dir, False, status)
            return output_dir, status
        adapter = (
            SyntheticAdapter()
            if config["source"] == "synthetic"
            else NativeAdapter(config["checkpoint"], config["device"])
        )
        if config["source"] == "synthetic":
            subjects, data_meta = synthetic_subjects(config)
        else:
            subjects, data_meta = native_subjects(config, adapter)
        if max(config["contexts"]) > adapter.context:
            raise ValueError("Requested context exceeds checkpoint context_max")
        context = dict(
            config=config, out=output_dir, adapter=adapter, subjects=subjects
        )
        if config["network_mapping"]:
            mapping = json.loads(
                Path(config["network_mapping"]).read_text(encoding="utf-8")
            )
            if not mapping.get("source") or not mapping.get("roi_to_network"):
                raise ValueError("Network mapping needs source and roi_to_network")
            if set(mapping["roi_to_network"]) - set(subjects[0].roi_labels):
                raise ValueError("Network mapping contains unknown ROI labels")
            context["networks"] = mapping["roi_to_network"]
            provenance["network_mapping"] = dict(
                source=mapping["source"], sha256=sha256(config["network_mapping"])
            )
        provenance.update(
            data=data_meta,
            model=adapter.metadata,
            n_subjects=len(subjects),
            split_counts={
                k: sum(s.split == k for s in subjects) for k in ("train", "val", "test")
            },
        )
        write_json(output_dir / "provenance.json", provenance)
        stages.write_table(
            output_dir,
            "subjects",
            [
                dict(
                    subject_id=s.subject_id,
                    diagnosis=s.diagnosis,
                    split=s.split,
                    n_tr=len(s.bold),
                    synthetic=s.synthetic,
                    **s.metadata,
                )
                for s in subjects
            ],
        )
        write_json(output_dir / "roi_labels.json", list(subjects[0].roi_labels))
        selected = STAGES if stage == "all" else (stage,)
        # Standalone clinical analysis computes necessary normative/errors too, transparently logged.
        if stage == "clinical":
            selected = ("performance", "normative", "clinical", "report")
        for name in selected:
            if name == "report":
                continue
            logging.info("Study stage %s started", name)
            try:
                result = (
                    (
                        "IMPLEMENTED_AND_TESTED",
                        "Frozen manifests, prefix scaling and checkpoint metadata recorded",
                    )
                    if name == "audit"
                    else getattr(stages, name)(context)
                )
                state, reason = result
                if state == "MISSING_DATA":
                    state, reason = "SKIPPED_MISSING_DATA", "MISSING_DATA: " + reason
                status[name] = dict(status=state, reason=reason)
            except Exception as exc:
                logging.exception("Stage %s failed", name)
                status[name] = dict(
                    status="FAILED", reason=f"{type(exc).__name__}: {exc}"
                )
            write_json(output_dir / "status.json", status)
            print(
                f"{name}: {status[name]['status']} | {status[name]['reason']}",
                flush=True,
            )
        if context.get("prediction_skips"):
            write_json(
                output_dir / "prediction_skips.json", context["prediction_skips"]
            )
        figures = render_report(output_dir, provenance["synthetic"], status)
        status["report"] = dict(
            status="IMPLEMENTED_AND_TESTED",
            reason=f"{len(figures)} table-driven figures (PNG/PDF)",
        )
        write_json(output_dir / "status.json", status)
    except Exception as exc:
        logging.exception("Study input/report failure")
        status["inputs_or_report"] = dict(
            status="FAILED", reason=f"{type(exc).__name__}: {exc}"
        )
        write_json(output_dir / "provenance.json", provenance)
        write_json(output_dir / "status.json", status)
        (output_dir / "report.md").write_text(
            "# Study failed\n\n"
            + status["inputs_or_report"]["reason"]
            + "\nSee run.log and status.json.\n",
            encoding="utf-8",
        )
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    return output_dir, status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument(
        "--input-run",
        type=Path,
        help="Re-render report from existing CSVs without rerunning inference",
    )
    args = parser.parse_args(argv)
    if args.input_run:
        if args.stage != "report":
            parser.error("--input-run requires --stage report")
        p = json.loads((args.input_run / "provenance.json").read_text(encoding="utf-8"))
        status = json.loads(
            (args.input_run / "status.json").read_text(encoding="utf-8")
        )
        render_report(args.input_run, p["synthetic"], status)
        return 0
    output_dir, status = run(load_config(args.config), args.stage)
    print(output_dir, flush=True)
    return int(any(v["status"] == "FAILED" for v in status.values()))


if __name__ == "__main__":
    raise SystemExit(main())
