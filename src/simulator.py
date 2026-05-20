"""End-to-end simulator (Plan §7).

Generates 1s base series for the whole market, applies anomaly scenarios,
aggregates dong-level meters, and exports TSB-AD-M format files.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .anomaly import (
    AnomalyInstance,
    apply_anomaly,
    load_scenarios,
    plan_anomalies,
)
from .baseline import CHANNELS, TimeAxis, env_series, store_baseline
from .exporter import write_tsb_format
from .topology import Market, build_market, summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("simulator")


@dataclass
class SimResult:
    market: Market
    ax: TimeAxis
    env: dict[str, np.ndarray]
    store_series: dict[str, dict[str, np.ndarray]]   # store_id -> channel -> array
    label_mask: dict[str, np.ndarray]                # store_id -> bool array
    nuisance_mask: dict[str, np.ndarray]             # store_id -> bool array
    panel_aggregates: dict[str, dict[str, np.ndarray]]   # panel_id -> ch -> array
    panel_label_mask: dict[str, np.ndarray]              # panel_id -> bool
    panel_nuisance_mask: dict[str, np.ndarray]
    dong_aggregates: dict[str, dict[str, np.ndarray]]    # dong_id -> ch -> array
    dong_label_mask: dict[str, np.ndarray]
    anomalies: list[AnomalyInstance]


def simulate(
    config_dir: str | Path,
    days: int = 30,
    seed: int = 42,
    base_freq_s: int = 1,
    inject_anomalies: bool = False,
) -> SimResult:
    config_dir = Path(config_dir)
    market = build_market(config_dir, seed=seed)
    log.info("market built: %s", summary(market))

    start = pd.Timestamp(market.start_date)
    n_seconds = days * 86400
    ax = TimeAxis(start=start, n_seconds=n_seconds, freq_s=base_freq_s)
    rng = np.random.default_rng(seed)

    env = env_series(market, ax, rng)
    log.info("env series ready (T mean=%.1f H mean=%.1f)",
             float(env["T_env"].mean()), float(env["H_env"].mean()))

    store_series: dict[str, dict[str, np.ndarray]] = {}
    label_mask: dict[str, np.ndarray] = {}
    nuisance_mask: dict[str, np.ndarray] = {}
    stores = list(market.stores)
    log.info("generating baseline for %d stores x %d ticks", len(stores), n_seconds)
    for i, s in enumerate(stores):
        store_series[s.store_id] = store_baseline(s, ax, env, market, rng)
        label_mask[s.store_id] = np.zeros(n_seconds, dtype=bool)
        nuisance_mask[s.store_id] = np.zeros(n_seconds, dtype=bool)
        if (i + 1) % 25 == 0:
            log.info("  baseline %d/%d", i + 1, len(stores))

    if inject_anomalies:
        scfg = load_scenarios(config_dir)
        anomalies = plan_anomalies(market, ax, scfg, rng)
        log.info("planned %d anomaly instances", len(anomalies))
        for ins in anomalies:
            store = market.store_by_id(ins.store_id)
            apply_anomaly(
                ins, store_series[ins.store_id], env, ax, store,
                label_mask[ins.store_id], nuisance_mask[ins.store_id],
            )
    else:
        anomalies = []
        log.info("anomaly injection disabled — clean baseline only")

    # ---- Panel (SDP) aggregates ----
    panel_agg: dict[str, dict[str, np.ndarray]] = {}
    panel_label: dict[str, np.ndarray] = {}
    panel_nui: dict[str, np.ndarray] = {}
    for panel in market.panels:
        sid_list = [s.store_id for s in panel.stores]
        if not sid_list:
            continue
        i_leak_panel = np.sum([store_series[sid]["I_leak"] for sid in sid_list], axis=0)
        i_load_panel = np.sum([store_series[sid]["I_load"] for sid in sid_list], axis=0)
        panel_agg[panel.panel_id] = {
            "I_leak_panel": i_leak_panel.astype(np.float32),
            "I_load_panel": i_load_panel.astype(np.float32),
        }
        # label union of stores in this panel
        lab = np.zeros(n_seconds, dtype=bool)
        nui = np.zeros(n_seconds, dtype=bool)
        for sid in sid_list:
            lab |= label_mask[sid]
            nui |= nuisance_mask[sid]
        panel_label[panel.panel_id] = lab
        panel_nui[panel.panel_id] = nui
    log.info("panel aggregates: %d panels", len(panel_agg))

    # ---- Dong-level (water main + MDP-side leak) ----
    dong_agg: dict[str, dict[str, np.ndarray]] = {}
    dong_label: dict[str, np.ndarray] = {}
    for d in market.dongs:
        sid_list = [s.store_id for s in d.stores]
        if not sid_list:
            continue
        q_main = np.sum([store_series[sid]["Q"] for sid in sid_list], axis=0)
        p_main = np.mean([store_series[sid]["P"] for sid in sid_list], axis=0)
        # 동의 SDP 누설전류 합 (MDP 측에서 본 신호)
        i_leak_dong = np.sum(
            [panel_agg[p.panel_id]["I_leak_panel"] for p in d.panels
             if p.panel_id in panel_agg], axis=0,
        )
        dong_agg[d.dong_id] = {
            "Q_main": q_main.astype(np.float32),
            "P_main": p_main.astype(np.float32),
            "I_leak_dong": i_leak_dong.astype(np.float32),
        }
        lab = np.zeros(n_seconds, dtype=bool)
        for sid in sid_list:
            lab |= label_mask[sid]
        dong_label[d.dong_id] = lab
    log.info("dong aggregates: %s", list(dong_agg.keys()))

    return SimResult(
        market=market, ax=ax, env=env,
        store_series=store_series,
        label_mask=label_mask, nuisance_mask=nuisance_mask,
        panel_aggregates=panel_agg,
        panel_label_mask=panel_label, panel_nuisance_mask=panel_nui,
        dong_aggregates=dong_agg, dong_label_mask=dong_label,
        anomalies=anomalies,
    )


# ---------- CLI ----------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Market AD Simulator")
    p.add_argument("--config", default="config/", help="config directory")
    p.add_argument("--out", default="data/", help="output directory")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--export-freqs", default="1m,30m,1h",
                   help="comma-separated freqs to export: 1s,30s,1m,30m,1h")
    p.add_argument("--store-level", action="store_true",
                   help="also export per-store 1s files (large)")
    p.add_argument("--store-sample", type=int, default=3,
                   help="when --store-level not set, dump N sample stores at 1s")
    p.add_argument("--inject-anomalies", action="store_true",
                   help="apply scenario-planned random anomalies (off by default)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    result = simulate(args.config, days=args.days, seed=args.seed,
                      inject_anomalies=args.inject_anomalies)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    freqs = _parse_freqs(args.export_freqs)
    log.info("exporting TSB-AD-M files: freqs=%s", freqs)
    write_tsb_format(
        result, out_dir=out, freqs_s=freqs,
        per_store_freq_s=1,
        per_store_ids=(None if args.store_level
                       else _sample_store_ids(result, args.store_sample)),
    )
    log.info("done. wrote to %s", out)


def _parse_freqs(spec: str) -> list[int]:
    out = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if tok.endswith("h"):
            out.append(int(tok[:-1]) * 3600)
        elif tok.endswith("m"):
            out.append(int(tok[:-1]) * 60)
        elif tok.endswith("s"):
            out.append(int(tok[:-1]))
        else:
            out.append(int(tok))
    return out


def _sample_store_ids(result: SimResult, k: int) -> list[str]:
    ids = [s.store_id for s in result.market.stores]
    # prefer stores with anomalies
    with_anom = sorted({a.store_id for a in result.anomalies})
    pick = with_anom[:k] if len(with_anom) >= k else (with_anom + ids[:k - len(with_anom)])
    return pick[:k]


if __name__ == "__main__":
    main()
