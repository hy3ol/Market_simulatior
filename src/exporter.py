"""TSB-AD-M format export (Plan §9).

Wide CSV: timestamp, ch_1, ch_2, ...
Companion label CSV: timestamp, label (0/1)
Meta JSON: anomaly index with scenario_type for nuisance filtering.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .simulator import SimResult


# ---------- helpers ----------

def _downsample(arr: np.ndarray, factor: int, agg: str = "mean") -> np.ndarray:
    if factor <= 1:
        return arr
    n = (len(arr) // factor) * factor
    arr = arr[:n].reshape(-1, factor)
    if agg == "mean":
        return arr.mean(axis=1).astype(np.float32)
    if agg == "max":
        return arr.max(axis=1).astype(np.float32)
    if agg == "any":
        return arr.any(axis=1)
    raise ValueError(agg)


def build_wide_market(result: "SimResult", freq_s: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Market-wide wide CSV (store x channel columns), plus per-row label."""
    full_idx = result.ax.index
    n = result.ax.n_seconds
    factor = freq_s // result.ax.freq_s

    out_idx = full_idx[::factor][:n // factor]
    cols: dict[str, np.ndarray] = {}
    label_arr = np.zeros(n, dtype=bool)
    # combine point labels per-store (any-OR)
    for sid, channels in result.store_series.items():
        for ch, arr in channels.items():
            cols[f"{sid}__{ch}"] = _downsample(arr, factor, agg="mean")
        label_arr |= result.label_mask[sid]
    # panel aggregates (SDP)
    for pid, chans in result.panel_aggregates.items():
        for ch, arr in chans.items():
            cols[f"{pid}__{ch}"] = _downsample(arr, factor, agg="mean")
    # dong aggregates
    for did, chans in result.dong_aggregates.items():
        for ch, arr in chans.items():
            cols[f"{did}__{ch}"] = _downsample(arr, factor, agg="mean")

    L = len(out_idx)
    # truncate any column length mismatches defensively
    for k, v in cols.items():
        if len(v) != L:
            cols[k] = v[:L]
    df = pd.DataFrame(cols, index=out_idx)
    df.index.name = "timestamp"

    lab_ds = _downsample(label_arr.astype(np.float32), factor, agg="max") > 0.5
    lab_df = pd.DataFrame({"label": lab_ds[:L].astype(int)}, index=out_idx)
    lab_df.index.name = "timestamp"
    return df, lab_df


def build_wide_store(result: "SimResult", sid: str, freq_s: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_idx = result.ax.index
    n = result.ax.n_seconds
    factor = freq_s // result.ax.freq_s
    out_idx = full_idx[::factor][:n // factor]
    cols = {ch: _downsample(arr, factor) for ch, arr in result.store_series[sid].items()}
    L = len(out_idx)
    df = pd.DataFrame({k: v[:L] for k, v in cols.items()}, index=out_idx)
    df.index.name = "timestamp"
    lab = _downsample(result.label_mask[sid].astype(np.float32), factor, agg="max") > 0.5
    nui = _downsample(result.nuisance_mask[sid].astype(np.float32), factor, agg="max") > 0.5
    lab_df = pd.DataFrame({
        "label": lab[:L].astype(int),
        "nuisance": nui[:L].astype(int),
    }, index=out_idx)
    lab_df.index.name = "timestamp"
    return df, lab_df


def _freq_tag(freq_s: int) -> str:
    if freq_s % 3600 == 0:
        return f"{freq_s // 3600}h"
    if freq_s % 60 == 0:
        return f"{freq_s // 60}m"
    return f"{freq_s}s"


# ---------- public API ----------

def write_tsb_format(
    result: "SimResult",
    out_dir: str | Path,
    freqs_s: list[int],
    per_store_freq_s: int = 1,
    per_store_ids: list[str] | None = None,
) -> dict:
    """Write all configured outputs. Returns a summary dict."""
    out_dir = Path(out_dir)
    tsb_dir = out_dir / "tsb_format"
    meta_dir = out_dir / "meta"
    tsb_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []

    # market-wide @ each freq
    for fs in freqs_s:
        tag = _freq_tag(fs)
        df, lab = build_wide_market(result, fs)
        stem = f"market_{tag}"
        df.to_csv(tsb_dir / f"{stem}.csv", float_format="%.4f")
        lab.to_csv(tsb_dir / f"{stem}.label.csv")
        written.append(stem)

    # per-store
    if per_store_ids is None:
        per_store_ids = [s.store_id for s in result.market.stores]
    for sid in per_store_ids:
        if sid not in result.store_series:
            continue
        df, lab = build_wide_store(result, sid, per_store_freq_s)
        tag = _freq_tag(per_store_freq_s)
        stem = f"store_{sid}_{tag}"
        df.to_csv(tsb_dir / f"{stem}.csv", float_format="%.4f")
        lab.to_csv(tsb_dir / f"{stem}.label.csv")
        written.append(stem)

    # anomaly index
    index = {
        "market": result.market.name,
        "start": str(result.ax.start),
        "n_seconds": result.ax.n_seconds,
        "base_freq_s": result.ax.freq_s,
        "instances": [a.to_dict() for a in result.anomalies],
    }
    with open(meta_dir / "anomaly_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2, default=_json_default)

    return {"written_stems": written, "n_instances": len(result.anomalies)}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f"not serializable: {type(o)}")
