"""Anomaly scenario library (Plan §6).

A1 drift, A2 spike, A3 intermittent, N1 inverter false-trip (nuisance).
Each scenario perturbs a single (store, channel) over a time window.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from .baseline import TimeAxis
from .topology import Market, Store


ScenarioType = Literal[
    "A1_drift", "A2_spike", "A3_intermittent", "N1_inverter_falsetrip",
    "A4_insulation_degradation", "A5_water_hammer",
    "A6_night_leak", "A7_load_leak_coupled",
]


@dataclass
class AnomalyInstance:
    id: str
    store_id: str
    channel: str
    scenario_type: ScenarioType
    start_idx: int            # 1s index
    end_idx: int              # inclusive
    severity: float
    label_kind: str
    extras: dict = field(default_factory=dict)  # e.g. point peak idx, breaker_off range

    def to_dict(self) -> dict:
        return asdict(self)


def load_scenarios(config_dir: str | Path) -> dict:
    with open(Path(config_dir) / "anomaly_scenarios.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- selection ----------

def _pick_scenario(cfg: dict, rng: np.random.Generator) -> tuple[str, dict]:
    items = list(cfg["scenarios"].items())
    weights = np.array([v["weight"] for _, v in items], dtype=float)
    weights /= weights.sum()
    idx = rng.choice(len(items), p=weights)
    return items[idx]


def _pick_channel(scfg: dict, rng: np.random.Generator) -> str:
    mix = scfg.get("channel_mix")
    if not mix:
        return rng.choice(scfg["target_channels"])
    chans = list(mix.keys())
    p = np.array([mix[c] for c in chans], dtype=float)
    p /= p.sum()
    return rng.choice(chans, p=p)


def _per_store_count(rng: np.random.Generator, dist: list[float]) -> int:
    return int(rng.choice(len(dist), p=np.array(dist) / np.sum(dist)))


def plan_anomalies(
    market: Market, ax: TimeAxis, cfg: dict, rng: np.random.Generator,
) -> list[AnomalyInstance]:
    """Draw a set of anomaly instances across all stores.

    Ensures min_instances_per_id is satisfied by padding scenario IDs that
    came up short.
    """
    instances: list[AnomalyInstance] = []
    next_id = [0]

    def _new_id(stype: str) -> str:
        next_id[0] += 1
        return f"{stype.split('_')[0]}-{next_id[0]:04d}"

    n = ax.n_seconds
    stores = list(market.stores)
    rng.shuffle(stores)

    # baseline draw — per-store count
    for s in stores:
        k = _per_store_count(rng, cfg["per_store_count_dist"])
        for _ in range(k):
            instances.append(_draw_one(s, ax, cfg, rng, _new_id))

    # ensure min instances per scenario id
    minimum = int(cfg.get("min_instances_per_id", 0))
    if minimum > 0:
        counts: dict[str, int] = {k: 0 for k in cfg["scenarios"]}
        for ins in instances:
            counts[ins.scenario_type] += 1
        for stype, c in counts.items():
            while c < minimum:
                # pick a random eligible store
                eligible = [s for s in stores if _eligible(s, stype, cfg)]
                if not eligible:
                    break
                s = eligible[int(rng.integers(0, len(eligible)))]
                instances.append(_draw_specific(s, stype, ax, cfg, rng, _new_id))
                c += 1
    return instances


def _eligible(store: Store, stype: str, cfg: dict) -> bool:
    scfg = cfg["scenarios"][stype]
    req = scfg.get("requires")
    if req == "has_inverter" and not store.profile.has_inverter:
        return False
    return True


def _draw_one(store, ax, cfg, rng, id_fn) -> AnomalyInstance:
    # weighted scenario pick, retry on eligibility
    for _ in range(6):
        stype, _ = _pick_scenario(cfg, rng)
        if _eligible(store, stype, cfg):
            return _draw_specific(store, stype, ax, cfg, rng, id_fn)
    # fallback — A2_spike always eligible
    return _draw_specific(store, "A2_spike", ax, cfg, rng, id_fn)


def _draw_specific(store, stype, ax, cfg, rng, id_fn) -> AnomalyInstance:
    scfg = cfg["scenarios"][stype]
    channel = _pick_channel(scfg, rng)
    severity = float(rng.uniform(*scfg["severity_range"]))
    n = ax.n_seconds

    if stype == "A1_drift":
        dur_min = rng.uniform(*scfg["duration_minutes_range"])
        dur = int(dur_min * 60)
        start = int(rng.integers(0, max(1, n - dur)))
        end = start + dur - 1
        return AnomalyInstance(
            id=id_fn(stype), store_id=store.store_id, channel=channel,
            scenario_type=stype, start_idx=start, end_idx=end,
            severity=severity, label_kind=scfg["label_kind"], extras={},
        )
    if stype == "A2_spike":
        dur_s = float(rng.uniform(*scfg["duration_seconds_range"]))
        dur = max(1, int(dur_s))
        start = int(rng.integers(0, max(1, n - dur)))
        end = start + dur - 1
        peak = int(rng.integers(start, end + 1))
        return AnomalyInstance(
            id=id_fn(stype), store_id=store.store_id, channel=channel,
            scenario_type=stype, start_idx=start, end_idx=end,
            severity=severity, label_kind=scfg["label_kind"],
            extras={"peak_idx": peak, "sigma_s": max(1.0, dur / 4.0)},
        )
    if stype == "A3_intermittent":
        # active window across multiple nights — pick a 7-day window
        window_days = int(rng.integers(1, 8))
        dur = window_days * 86400
        start = int(rng.integers(0, max(1, n - dur)))
        end = min(n - 1, start + dur - 1)
        return AnomalyInstance(
            id=id_fn(stype), store_id=store.store_id, channel=channel,
            scenario_type=stype, start_idx=start, end_idx=end,
            severity=severity, label_kind=scfg["label_kind"],
            extras={
                "hour_window": scfg["gating"]["hour_window"],
                "humidity_min": scfg["gating"]["humidity_min"],
            },
        )
    if stype == "N1_inverter_falsetrip":
        dur = max(1, int(rng.uniform(*scfg["duration_seconds_range"])))
        start = int(rng.integers(0, max(1, n - dur - scfg["breaker_off_minutes"] * 60)))
        end = start + dur - 1
        breaker_off_end = end + scfg["breaker_off_minutes"] * 60
        return AnomalyInstance(
            id=id_fn(stype), store_id=store.store_id, channel=channel,
            scenario_type=stype, start_idx=start, end_idx=end,
            severity=severity, label_kind=scfg["label_kind"],
            extras={"breaker_off_end": breaker_off_end,
                    "peak_idx": int((start + end) // 2)},
        )
    if stype == "A4_insulation_degradation":
        dur_min = rng.uniform(*scfg["duration_minutes_range"])
        dur = int(dur_min * 60)
        start = int(rng.integers(0, max(1, n - dur)))
        end = start + dur - 1
        n_steps = int(rng.integers(*scfg["n_steps_range"]))
        # step boundaries within [start, end] — sorted
        step_offsets = sorted(rng.integers(0, dur, size=n_steps).tolist())
        return AnomalyInstance(
            id=id_fn(stype), store_id=store.store_id, channel=channel,
            scenario_type=stype, start_idx=start, end_idx=end,
            severity=severity, label_kind=scfg["label_kind"],
            extras={"step_offsets": step_offsets},
        )
    if stype == "A5_water_hammer":
        dur = max(2, int(rng.uniform(*scfg["duration_seconds_range"])))
        start = int(rng.integers(0, max(1, n - dur)))
        end = start + dur - 1
        n_osc = int(rng.integers(*scfg["n_oscillations_range"]))
        return AnomalyInstance(
            id=id_fn(stype), store_id=store.store_id, channel=channel,
            scenario_type=stype, start_idx=start, end_idx=end,
            severity=severity, label_kind=scfg["label_kind"],
            extras={"n_osc": n_osc, "peak_idx": start + dur // 4},
        )
    if stype == "A6_night_leak":
        dur_min = rng.uniform(*scfg["duration_minutes_range"])
        dur = int(dur_min * 60)
        # 시작은 야간 시간대로 강제 (22~02시)
        # 임의 day_offset 잡고 그 날 23시로 align
        day_idx = int(rng.integers(0, max(1, n // 86400)))
        start = day_idx * 86400 + 23 * 3600
        start = min(start, n - dur - 1)
        end = min(n - 1, start + dur - 1)
        return AnomalyInstance(
            id=id_fn(stype), store_id=store.store_id, channel=channel,
            scenario_type=stype, start_idx=start, end_idx=end,
            severity=severity, label_kind=scfg["label_kind"],
            extras={"hour_window": scfg["gating"]["hour_window"]},
        )
    if stype == "A7_load_leak_coupled":
        dur_min = rng.uniform(*scfg["duration_minutes_range"])
        dur = int(dur_min * 60)
        start = int(rng.integers(0, max(1, n - dur)))
        end = start + dur - 1
        return AnomalyInstance(
            id=id_fn(stype), store_id=store.store_id, channel=channel,
            scenario_type=stype, start_idx=start, end_idx=end,
            severity=severity, label_kind=scfg["label_kind"], extras={},
        )
    raise ValueError(stype)


# ---------- perturbation ----------

def apply_anomaly(
    ins: AnomalyInstance,
    series: dict[str, np.ndarray],
    env: dict[str, np.ndarray],
    ax: TimeAxis,
    store: Store,
    label_mask: np.ndarray,
    nuisance_mask: np.ndarray,
) -> None:
    """Mutate `series[channel]` in-place; mark label_mask/nuisance_mask."""
    n = ax.n_seconds
    ch = ins.channel
    if ch not in series:
        return
    arr = series[ch]
    s, e = ins.start_idx, min(ins.end_idx, n - 1)

    if ins.scenario_type == "A1_drift":
        t = np.arange(e - s + 1, dtype=np.float32) / 3600.0  # hours since start
        epsilon = ins.severity / 100.0  # fraction/hour
        base_at_s = float(np.mean(arr[s:e + 1])) + 1e-6
        delta = base_at_s * epsilon * t
        arr[s:e + 1] = arr[s:e + 1] + delta
        label_mask[s:e + 1] = True

    elif ins.scenario_type == "A2_spike":
        peak = ins.extras["peak_idx"]
        sigma = ins.extras["sigma_s"]
        idx = np.arange(s, e + 1)
        gauss = np.exp(-0.5 * ((idx - peak) / max(sigma, 0.5)) ** 2).astype(np.float32)
        base_at_peak = float(arr[peak]) + 1e-6
        arr[s:e + 1] = arr[s:e + 1] + base_at_peak * ins.severity * gauss
        # point label = peak; range label = [s,e]
        label_mask[s:e + 1] = True

    elif ins.scenario_type == "A3_intermittent":
        h0, h1 = ins.extras["hour_window"]
        hum_min = ins.extras["humidity_min"]
        hours = ax.hours[s:e + 1]
        if h0 <= h1:
            in_window = (hours >= h0) & (hours < h1)
        else:  # wraps midnight
            in_window = (hours >= h0) | (hours < h1)
        gated = in_window & (env["H_env"][s:e + 1] >= hum_min)
        if not gated.any():
            return
        base = float(np.mean(arr[s:e + 1])) + 1e-6
        bump = base * ins.severity * gated.astype(np.float32)
        arr[s:e + 1] = arr[s:e + 1] + bump
        full = np.zeros(n, dtype=bool)
        full[s:e + 1] = gated
        label_mask |= full

    elif ins.scenario_type == "N1_inverter_falsetrip":
        peak = ins.extras["peak_idx"]
        idx = np.arange(s, e + 1)
        sigma = max(1.0, (e - s) / 3.0)
        gauss = np.exp(-0.5 * ((idx - peak) / sigma) ** 2).astype(np.float32)
        base_at_peak = float(arr[peak]) + 1e-6
        arr[s:e + 1] = arr[s:e + 1] + base_at_peak * ins.severity * gauss
        # also boost I_leak briefly
        if "I_leak" in series and ch != "I_leak":
            series["I_leak"][s:e + 1] = series["I_leak"][s:e + 1] + \
                float(np.mean(series["I_leak"][s:e + 1])) * 1.5 * gauss
        # breaker OFF — kill I_load and Q to ~0
        bo_end = min(n - 1, ins.extras["breaker_off_end"])
        if "I_load" in series:
            series["I_load"][e + 1:bo_end + 1] = 0.05
        if "Q" in series:
            series["Q"][e + 1:bo_end + 1] = 0.0
        label_mask[s:e + 1] = True
        nuisance_mask[s:e + 1] = True

    elif ins.scenario_type == "A4_insulation_degradation":
        # 점진적 절연 열화: linear ramp + step-wise jumps
        L = e - s + 1
        t = np.arange(L, dtype=np.float32) / L
        base = float(np.median(arr[s:e + 1])) + 1e-6
        ramp = ins.severity * t * 0.6   # 최종 증가량의 60%는 ramp
        # step-wise jumps 누적
        steps = np.zeros(L, dtype=np.float32)
        for off in ins.extras.get("step_offsets", []):
            off = int(off)
            if 0 <= off < L:
                steps[off:] += ins.severity * 0.4 / max(1, len(ins.extras["step_offsets"]))
        arr[s:e + 1] = arr[s:e + 1] + ramp + steps
        label_mask[s:e + 1] = True

    elif ins.scenario_type == "A5_water_hammer":
        # damped oscillation on P (or target channel)
        L = e - s + 1
        t = np.arange(L, dtype=np.float32)
        n_osc = max(1, int(ins.extras.get("n_osc", 4)))
        freq = n_osc / max(1.0, L)
        damp = np.exp(-3.0 * t / max(1.0, L)).astype(np.float32)
        osc = np.sin(2 * np.pi * freq * t).astype(np.float32) * damp
        arr[s:e + 1] = arr[s:e + 1] + ins.severity * osc
        # 동반: Q는 짧은 dip (밸브 닫힘 효과)
        if "Q" in series and ch != "Q":
            series["Q"][s:e + 1] = np.maximum(
                0.0, series["Q"][s:e + 1] - 0.5 * np.abs(osc) * float(np.mean(series["Q"][s:e + 1]) + 0.1)
            )
        label_mask[s:e + 1] = True

    elif ins.scenario_type == "A6_night_leak":
        h0, h1 = ins.extras["hour_window"]
        hours = ax.hours[s:e + 1]
        if h0 <= h1:
            in_window = (hours >= h0) & (hours < h1)
        else:
            in_window = (hours >= h0) | (hours < h1)
        # ramp up 점진 증가
        ramp = np.linspace(0, 1, e - s + 1, dtype=np.float32)
        bump = ins.severity * ramp * in_window.astype(np.float32)
        arr[s:e + 1] = arr[s:e + 1] + bump
        full = np.zeros(n, dtype=bool)
        full[s:e + 1] = in_window & (bump > 0.05)
        label_mask |= full

    elif ins.scenario_type == "A7_load_leak_coupled":
        # I_leak이 I_load에 비례하여 추가 상승
        if "I_load" not in series:
            return
        load = series["I_load"][s:e + 1]
        load_norm = load / (float(np.max(load)) + 1e-6)
        # severity = mA per (normalized load) — 부하 동작 시에만 강하게 가산
        bump = ins.severity * (load_norm ** 1.3)
        arr[s:e + 1] = arr[s:e + 1] + bump
        label_mask[s:e + 1] = True
