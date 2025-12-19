#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced analysis for Scheduling results: computes MAE, MAPE, SMAPE, RMSE,
bootstrap CIs, selection metrics, and generates figures for a paper report.

Usage:
  python analysis_scheduling_results.py --root /path/to/Results --out /path/to/out_dir

"""
from __future__ import annotations
import argparse
import re
import math
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse much of the parsing logic from your template

@dataclass(frozen=True)
class FileMeta:
    file_path: str
    family: str
    model: Optional[str]
    metric: Optional[str]
    src: Optional[int]
    dst: Optional[int]
    split: Optional[str]
    topology: Optional[str]
    traffic_model: Optional[str]
    scheduler: Optional[str]
    raw_name: str


def _safe_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def parse_filename_meta(path: Path) -> FileMeta:
    name = path.name
    family = path.parent.name

    model = None
    metric = None
    src = None
    dst = None
    split = None
    topology = None
    traffic_model = None
    scheduler = None

    lower = name.lower()

    if "fermi" in family.lower() or "_fermi" in lower:
        model = "fermi"
    if "erlang" in family.lower() or "_erlang" in lower:
        model = "erlang"

    m_metric = re.search(r"(?:_|processed_)(delay|jitter|loss)(?:_|\.txt)", lower)
    if m_metric:
        metric = m_metric.group(1)

    m_pair = re.search(r"(?:candidate_routes|trafficmodels_results)_(\d+)_(\d+)_", lower)
    if m_pair:
        src = _safe_int(m_pair.group(1))
        dst = _safe_int(m_pair.group(2))

    m_split_topo = re.search(r"_(train|test)_(gbn|geant2|nsfnet|rediris)_", lower)
    if m_split_topo:
        split = m_split_topo.group(1)
        topology = m_split_topo.group(2)

    m_sched = re.search(r"_(wfq(?:-drr-sp)?)[-_]", lower)
    if m_sched:
        scheduler = m_sched.group(1)

    return FileMeta(
        file_path=str(path),
        family=family,
        model=model,
        metric=metric,
        src=src,
        dst=dst,
        split=split,
        topology=topology,
        traffic_model=traffic_model,
        scheduler=scheduler,
        raw_name=name,
    )


# Regexes for parsing
CAND_HDR_RE = re.compile(
    r"^\[(?P<brand>[A-Za-z]+)\]\s+Candidate\s+(?P<cand>\d+):\s+src=(?P<src>\d+),\s+dst=(?P<dst>\d+),\s+path\s*=\s*(?P<path>[0-9]+(?:->[0-9]+)*)\s*$"
)

ROW_RE = re.compile(
    r"^\s*(?P<idx>\d+)\s+"
    r"(?P<src>\d+)\s+"
    r"(?P<dst>\d+)\s*(?P<star>\*)?\s+"
    r"(?P<route>[0-9]+(?:->[0-9]+)*)\s+"
    r"(?P<traffic>[+-]?\d+(?:\.\d+)?)\s+"
    r"(?P<packets>[+-]?\d+(?:\.\d+)?)\s+"
    r"(?P<true>[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s+"
    r"(?P<pred>[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*$",
    re.IGNORECASE,
)

MODE_RE = re.compile(r"^===\s*(?P<mode>.+?)\s*\(metric=(?P<metric>[a-z_]+)\)\s*===\s*$", re.IGNORECASE)

TF_NOISE_PREFIXES = ("WARNING:tensorflow",)
TF_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+:\s+[IWE]\s+tensorflow/")


def parse_log_file(path: Path, meta: FileMeta) -> Tuple[pd.DataFrame, Dict]:
    text = path.read_text(errors="replace").splitlines()

    run_info: Dict[str, Optional[str]] = {
        "declared_metric": None,
        "declared_mode_line": None,
        "checkpoint": None,
        "dataset_dir": None,
        "sample_index": None,
        "routes_file": None,
        "best_checkpoint": None,
        "n_paths": None,
    }

    cleaned: List[str] = []
    for line in text:
        if TF_LINE_RE.match(line) or line.startswith(TF_NOISE_PREFIXES):
            continue
        cleaned.append(line.rstrip("\n"))

    for line in cleaned:
        m = MODE_RE.match(line.strip())
        if m:
            run_info["declared_mode_line"] = m.group("mode")
            run_info["declared_metric"] = m.group("metric").lower()
        if line.lower().startswith("dataset directory"):
            run_info["dataset_dir"] = line.split(":", 1)[-1].strip()
        if line.lower().startswith("checkpoint dir"):
            run_info["checkpoint"] = line.split(":", 1)[-1].strip()
        if line.lower().startswith("sample index"):
            run_info["sample_index"] = line.split(":", 1)[-1].strip()
        if line.lower().startswith("routes file"):
            run_info["routes_file"] = line.split(":", 1)[-1].strip()
        if "best checkpoint found" in line.lower():
            run_info["best_checkpoint"] = line.split(":", 1)[-1].strip()
        if "loaded baseline sample" in line.lower() and "n_paths" in line.lower():
            mnp = re.search(r"n_paths\s*=\s*(\d+)", line.lower())
            if mnp:
                run_info["n_paths"] = mnp.group(1)

    current_cand = None
    current_path = None
    current_brand = None

    records = []

    for line in cleaned:
        s = line.strip()
        mh = CAND_HDR_RE.match(s)
        if mh:
            current_brand = mh.group("brand").lower()
            current_cand = int(mh.group("cand"))
            current_path = mh.group("path")
            continue

        mr = ROW_RE.match(line)
        if mr and current_cand is not None:
            truev = float(mr.group("true"))
            predv = float(mr.group("pred"))
            err = predv - truev
            abs_err = abs(err)
            perc_err = (abs_err / abs(truev) * 100.0) if truev != 0 else np.nan
            smape = 2.0 * abs_err / (abs(truev) + abs(predv)) * 100.0 if (abs(truev) + abs(predv)) != 0 else np.nan

            records.append(
                {
                    "candidate_id": current_cand,
                    "candidate_path": current_path,
                    "brand_in_body": current_brand,
                    "row_idx": int(mr.group("idx")),
                    "od_src": int(mr.group("src")),
                    "od_dst": int(mr.group("dst")),
                    "is_target_row": True if mr.group("star") else False,
                    "route_nodes": mr.group("route"),
                    "traffic": float(mr.group("traffic")),
                    "packets": float(mr.group("packets")),
                    "true_value": truev,
                    "pred_value": predv,
                    "error": err,
                    "abs_error": abs_err,
                    "sq_error": err * err,
                    "perc_error": perc_err,
                    "smape": smape,
                }
            )

    rows_df = pd.DataFrame.from_records(records)

    if not rows_df.empty:
        for k, v in asdict(meta).items():
            rows_df[k] = v
        for k, v in run_info.items():
            rows_df[k] = v

    return rows_df, run_info


# selection metrics same as template

def selection_metrics(rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame()

    target = rows_df[rows_df["is_target_row"] == True].copy()
    if target.empty:
        return pd.DataFrame()

    key_cols = ["file_path", "family", "model", "metric", "src", "dst", "split", "topology", "traffic_model", "scheduler"]
    grp = target.groupby(key_cols, dropna=False)

    out = []
    for key, g in grp:
        g_sorted_pred = g.sort_values("pred_value", ascending=True)
        g_sorted_true = g.sort_values("true_value", ascending=True)

        chosen = g_sorted_pred.iloc[0]
        oracle = g_sorted_true.iloc[0]

        regret = float(chosen["true_value"] - oracle["true_value"])
        top1 = 1 if int(chosen["candidate_id"]) == int(oracle["candidate_id"]) else 0

        pred_rank = g["pred_value"].rank(method="average")
        true_rank = g["true_value"].rank(method="average")
        if len(g) >= 2:
            rho = float(np.corrcoef(pred_rank.values, true_rank.values)[0, 1])
        else:
            rho = np.nan

        out.append({
            **{col: val for col, val in zip(key_cols, key)},
            "n_candidates": int(g.shape[0]),
            "chosen_candidate_id": int(chosen["candidate_id"]),
            "oracle_candidate_id": int(oracle["candidate_id"]),
            "chosen_true": float(chosen["true_value"]),
            "oracle_true": float(oracle["true_value"]),
            "regret": regret,
            "top1": top1,
            "spearman_rho_candidates": rho,
        })

    return pd.DataFrame(out)


# bootstrap and accuracy summary with MAPE/SMAPE

def bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int = 2000, alpha: float = 0.05, seed: int = 7):
    rng = np.random.default_rng(seed)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    stats = []
    n = len(values)
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        stats.append(stat_fn(sample))
    stats = np.array(stats)
    lo = np.quantile(stats, alpha / 2)
    hi = np.quantile(stats, 1 - alpha / 2)
    return (float(np.mean(stats)), float(lo), float(hi))


def accuracy_summary(rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame()

    target = rows_df[rows_df["is_target_row"] == True].copy()
    if target.empty:
        return pd.DataFrame()

    group_cols = ["family", "model", "metric", "split", "topology", "traffic_model", "scheduler"]
    out = []

    for key, g in target.groupby(group_cols, dropna=False):
        e = g["error"].to_numpy()
        ae = g["abs_error"].to_numpy()
        se = g["sq_error"].to_numpy()
        pe = g["perc_error"].to_numpy()
        sm = g["smape"].to_numpy()

        mae = float(np.nanmean(ae))
        rmse = float(math.sqrt(np.nanmean(se)))
        bias = float(np.nanmean(e))
        med_ae = float(np.nanmedian(ae))
        p95_ae = float(np.nanquantile(ae, 0.95))

        # MAPE (exclude NaNs produced by true==0)
        mape = float(np.nanmean(pe))
        med_mape = float(np.nanmedian(pe)) if np.any(~np.isnan(pe)) else np.nan
        p95_mape = float(np.nanquantile(pe[~np.isnan(pe)], 0.95)) if np.any(~np.isnan(pe)) else np.nan

        # SMAPE
        smape = float(np.nanmean(sm)) if np.any(~np.isnan(sm)) else np.nan

        mae_mean, mae_lo, mae_hi = bootstrap_ci(ae, np.mean)
        rmse_mean, rmse_lo, rmse_hi = bootstrap_ci(np.sqrt(se), np.mean)
        mape_mean, mape_lo, mape_hi = bootstrap_ci(pe[~np.isnan(pe)] if np.any(~np.isnan(pe)) else np.array([]), np.mean)

        if g.shape[0] >= 2:
            r = float(np.corrcoef(g["true_value"], g["pred_value"])[0, 1])
        else:
            r = np.nan

        out.append({
            **{col: val for col, val in zip(group_cols, key)},
            "n_points": int(g.shape[0]),
            "MAE": mae,
            "RMSE": rmse,
            "Bias": bias,
            "MedianAE": med_ae,
            "P95AE": p95_ae,
            "MAPE": mape,
            "MedianMAPE": med_mape,
            "P95MAPE": p95_mape,
            "SMAPE": smape,
            "MAE_boot_mean": mae_mean,
            "MAE_CI95_lo": mae_lo,
            "MAE_CI95_hi": mae_hi,
            "MAPE_boot_mean": mape_mean,
            "MAPE_CI95_lo": mape_lo,
            "MAPE_CI95_hi": mape_hi,
            "RMSE_boot_mean": rmse_mean,
            "RMSE_CI95_lo": rmse_lo,
            "RMSE_CI95_hi": rmse_hi,
            "Pearson_r_true_pred": r,
        })

    return pd.DataFrame(out)


# extra comparison summary

def model_comparison_table(acc_df: pd.DataFrame) -> pd.DataFrame:
    # pivot to compare fermi vs erlang for each group
    # find matching groups where both models present
    comps = []
    by_key = ["metric", "split", "topology", "scheduler"]
    grp = acc_df.groupby(by_key, dropna=False)
    for key, g in grp:
        # require both models
        if not set(g["model"]) >= {"fermi", "erlang"}:
            continue
        row_fermi = g[g["model"] == "fermi"].iloc[0]
        row_erlang = g[g["model"] == "erlang"].iloc[0]

        delta_mae = row_fermi["MAE"] - row_erlang["MAE"]
        pct_improve_mae = (delta_mae / row_erlang["MAE"] * 100.0) if row_erlang["MAE"] != 0 else np.nan
        delta_mape = row_fermi["MAPE"] - row_erlang["MAPE"]
        pct_improve_mape = (delta_mape / row_erlang["MAPE"] * 100.0) if row_erlang["MAPE"] != 0 else np.nan

        comps.append({
            "metric": key[0],
            "split": key[1],
            "topology": key[2],
            "scheduler": key[3],
            "fermi_MAE": float(row_fermi["MAE"]),
            "erlang_MAE": float(row_erlang["MAE"]),
            "delta_MAE": float(delta_mae),
            "pct_diff_MAE": float(pct_improve_mae),
            "fermi_MAPE": float(row_fermi["MAPE"]),
            "erlang_MAPE": float(row_erlang["MAPE"]),
            "delta_MAPE": float(delta_mape),
            "pct_diff_MAPE": float(pct_improve_mape),
        })

    return pd.DataFrame(comps)


# plots

def make_plots(rows: pd.DataFrame, acc: pd.DataFrame, sel: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    target = rows[rows["is_target_row"] == True].copy()

    # Scatter true vs pred per model/metric
    for (metric, model), g in target.groupby(["metric", "model"], dropna=False):
        if metric is None or model is None:
            continue
        plt.figure(figsize=(4, 4))
        plt.scatter(g["true_value"], g["pred_value"], s=8, alpha=0.6)
        m = max(max(g["true_value"].max(), g["pred_value"].max()), 1.0)
        plt.plot([0, m], [0, m], color="k", linestyle="--", linewidth=1)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"True vs Pred ({metric}) - {model}")
        plt.tight_layout()
        plt.savefig(out_dir / f"scatter_true_vs_pred_{model}_{metric}.png", dpi=200)
        plt.close()

    # Boxplot of percent error by model for each metric
    for metric, g in target.groupby("metric", dropna=False):
        if metric is None:
            continue
        plt.figure(figsize=(6, 4))
        data = [grp["perc_error"].dropna().values for _, grp in g.groupby("model", dropna=False)]
        labels = [str(m) for m, _ in g.groupby("model", dropna=False)]
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.ylabel("Absolute % Error")
        plt.title(f"Absolute % Error by model ({metric})")
        plt.tight_layout()
        plt.savefig(out_dir / f"box_perc_error_by_model_{metric}.png", dpi=200)
        plt.close()

    # Bar plots: MAE and MAPE per model grouped by metric
    if not acc.empty:
        for metric, g in acc.groupby("metric", dropna=False):
            plt.figure(figsize=(6, 4))
            x = np.arange(len(g))
            labels = [f"{row['model']}\n{row['topology']}\n{row['scheduler']}" for _, row in g.iterrows()]
            maes = g["MAE"].values
            mapes = g["MAPE"].values
            plt.bar(x - 0.15, maes, width=0.3, label="MAE")
            plt.bar(x + 0.15, mapes, width=0.3, label="MAPE")
            plt.xticks(x, labels, rotation=45, ha="right")
            plt.ylabel("Error")
            plt.title(f"MAE and MAPE across groups ({metric})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"bar_mae_mape_{metric}.png", dpi=200)
            plt.close()

    # Regret histogram
    if not sel.empty:
        plt.figure(figsize=(5, 4))
        plt.hist(sel["regret"].dropna(), bins=40)
        plt.xlabel("Regret (true(chosen) - true(oracle))")
        plt.ylabel("Count")
        plt.title("Candidate-selection regret distribution")
        plt.tight_layout()
        plt.savefig(out_dir / "hist_regret.png", dpi=200)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory containing the experiment folders")
    ap.add_argument("--out", required=True, help="Output directory for csv + figures")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted([p for p in root.rglob("*.txt") if p.is_file()])
    if not txt_files:
        raise SystemExit(f"No .txt files found under: {root}")

    all_rows = []
    all_meta = []
    for p in txt_files:
        meta = parse_filename_meta(p)
        all_meta.append(asdict(meta))
        rows_df, run_info = parse_log_file(p, meta)
        if not rows_df.empty:
            all_rows.append(rows_df)

    meta_df = pd.DataFrame(all_meta)
    meta_df.to_csv(out_dir / "file_index.csv", index=False)

    if not all_rows:
        print("Parsed 0 candidate rows. Check that logs contain '[X] Candidate' blocks + tables.")
        return

    rows = pd.concat(all_rows, ignore_index=True)
    try:
        rows.to_parquet(out_dir / "rows.parquet", index=False)
    except Exception as e:
        print("Warning: could not write parquet file:", e)
    rows.to_csv(out_dir / "rows.csv", index=False)

    sel = selection_metrics(rows)
    sel.to_csv(out_dir / "candidate_selection.csv", index=False)

    acc = accuracy_summary(rows)
    acc.to_csv(out_dir / "accuracy_summary.csv", index=False)

    comps = model_comparison_table(acc)
    comps.to_csv(out_dir / "model_comparison.csv", index=False)

    make_plots(rows, acc, sel, out_dir / "figures")

    snapshot = {
        "n_files_total": len(txt_files),
        "n_rows_parsed": int(rows.shape[0]),
        "n_target_rows": int((rows["is_target_row"] == True).sum()),
        "n_candidate_sets": int(sel.shape[0]),
    }
    (out_dir / "snapshot.json").write_text(json.dumps(snapshot, indent=2))

    print("Done.")
    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()
