#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract candidate paths and count how often each candidate path was the best (lowest true delay/jitter).
Writes:
 - candidates_all.csv  (one row per candidate target row)
 - best_counts_delay.csv, best_counts_jitter.csv  (counts per candidate_path)
 - best_counts_summary.csv (combined)
 - figures/bar_best_counts_{metric}.png

Usage:
  python extract_candidate_best_counts.py --results-root ./Results --out ./analysis_results/extract

"""
from __future__ import annotations
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

# Regexes (same as analysis script)
CAND_HDR_RE = re.compile(
    r"^\[(?P<brand>[A-Za-z-]+)\]\s+Candidate\s+(?P<cand>\d+):\s+src=(?P<src>\d+),\s+dst=(?P<dst>\d+),\s+path\s*=\s*(?P<path>[0-9]+(?:->[0-9]+)*)\s*$"
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
TF_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def parse_file(path: Path) -> List[Dict]:
    text = path.read_text(errors='replace').splitlines()
    cleaned = [l for l in text if not TF_LINE_RE.match(l)]

    metric = None
    # try to get metric from header
    for l in cleaned:
        m = MODE_RE.match(l.strip())
        if m:
            metric = m.group('metric').lower()
            break
    # fallback: look in filename
    if metric is None:
        lower = path.name.lower()
        if 'delay' in lower:
            metric = 'delay'
        elif 'jitter' in lower:
            metric = 'jitter'
        elif 'loss' in lower:
            metric = 'loss'

    # extract graph
    graph = None
    import re
    m = re.search(r'_Scheduling_([^_]+)_erlang', path.name)
    if m:
        graph = m.group(1)
    else:
        m = re.search(r'_TrafficModels_([^_]+)_fermi', path.name)
        if m:
            graph = m.group(1)

    records = []
    current_cand = None
    current_path = None
    current_brand = None

    for line in cleaned:
        s = line.strip()
        mh = CAND_HDR_RE.match(s)
        if mh:
            current_brand = mh.group('brand').lower()
            current_cand = int(mh.group('cand'))
            current_path = mh.group('path')
            continue
        mr = ROW_RE.match(line)
        if mr and current_cand is not None:
            is_target = True if mr.group('star') else False
            if not is_target:
                continue
            truev = float(mr.group('true'))
            predv = float(mr.group('pred'))
            records.append({
                'file': str(path),
                'metric': metric,
                'graph': graph,
                'candidate_id': current_cand,
                'candidate_path': current_path,
                'true_value': truev,
                'pred_value': predv,
                'od_src': int(mr.group('src')),
                'od_dst': int(mr.group('dst')),
                'brand': current_brand,
            })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-root', default='./Results')
    ap.add_argument('--out', default='./analysis_results/extract')
    args = ap.parse_args()

    root = Path(args.results_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(root.rglob('*.txt'))
    if not txt_files:
        raise SystemExit(f'No txt files under {root}')

    all_recs = []
    for p in txt_files:
        recs = parse_file(p)
        if recs:
            all_recs.extend(recs)

    if not all_recs:
        print('No target rows parsed.')
        return

    df = pd.DataFrame(all_recs)
    df.to_csv(out / 'candidates_all.csv', index=False)

    summaries = []
    for metric in ['delay', 'jitter']:
        dfm = df[df['metric'] == metric].copy()
        if dfm.empty:
            continue
        # overall
        winners = []
        for fname, g in dfm.groupby('file'):
            idx = g['true_value'].idxmin()
            winners.append(g.loc[idx])
        winners_df = pd.DataFrame(winners)
        counts = winners_df.groupby(['candidate_path']).size().reset_index(name='count')
        counts = counts.sort_values('count', ascending=False)
        counts.to_csv(out / f'best_counts_{metric}.csv', index=False)

        # per graph
        for graph_name, gdf in dfm.groupby('graph'):
            if gdf.empty:
                continue
            winners_g = []
            for fname, gg in gdf.groupby('file'):
                idx = gg['true_value'].idxmin()
                winners_g.append(gg.loc[idx])
            winners_g_df = pd.DataFrame(winners_g)
            counts_g = winners_g_df.groupby(['candidate_path']).size().reset_index(name='count')
            counts_g = counts_g.sort_values('count', ascending=False)
            counts_g.to_csv(out / f'best_counts_{metric}_{graph_name}.csv', index=False)

        # top candidates
        summaries.append({'metric': metric, 'n_files': int(dfm['file'].nunique()), 'n_winners': int(counts.shape[0])})

        # plot top 20
        top = counts.head(20)
        # plt.figure(figsize=(8, max(3, 0.3*len(top))))
        # plt.barh(range(len(top)), top['count'][::-1])
        # plt.yticks(range(len(top)), top['candidate_path'][::-1])
        # plt.xlabel('Times best (oracle)')
        # plt.title(f'Top candidate paths by times being best ({metric})')
        # plt.tight_layout()
        # plt.savefig(out / f'bar_best_counts_{metric}.png', dpi=200)
        # plt.close()

    pd.DataFrame(summaries).to_csv(out / 'best_counts_summary.csv', index=False)

    print('Wrote results to', out)

if __name__ == '__main__':
    main()
