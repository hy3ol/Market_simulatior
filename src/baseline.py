"""Baseline signal synthesis (Plan §5).

Produces 1s base series for each (store, channel). Vectorized over time.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .topology import Market, Store


CHANNELS = ["I_leak", "I_load", "Q", "P", "T_env", "H_env"]


@dataclass
class TimeAxis:
    start: pd.Timestamp
    n_seconds: int
    freq_s: int = 1

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, periods=self.n_seconds, freq=f"{self.freq_s}s")

    @property
    def hours(self) -> np.ndarray:
        # fractional hour-of-day for each tick
        idx = self.index
        return idx.hour.values + idx.minute.values / 60.0 + idx.second.values / 3600.0

    @property
    def day_of_week(self) -> np.ndarray:
        return self.index.dayofweek.values

    @property
    def day_of_month(self) -> np.ndarray:
        return (self.index.day - 1).values  # 0-indexed

    @property
    def month(self) -> np.ndarray:
        return self.index.month.values


# ---------- environment (shared across stores) ----------

def _slow_drift_centered(n: int, sigma_frac: float, rng: np.random.Generator,
                          knot_period_s: int = 3600) -> np.ndarray:
    """Slow random walk centered at 0. 한 knot/시간 단위, 며칠 스케일 weather front 시뮬."""
    n_knots = max(8, n // knot_period_s)
    knots = rng.normal(0, sigma_frac, size=n_knots).cumsum()
    knots = knots - knots.mean()
    xp = np.linspace(0, n - 1, n_knots)
    return np.interp(np.arange(n), xp, knots).astype(np.float32)


def env_series(market: Market, ax: TimeAxis, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """기온/습도. 일교차(diurnal) + 며칠 단위 weather front(slow drift) + 비 에피소드(부드러운 ramp).
    - 기온은 ~15h에 피크, 습도는 ~5h에 피크 (메테오로지 기본)
    - 두 변수는 일교차에선 반비례하지만, 독립적인 슬로우 드리프트로 완벽한 거울 패턴은 깨짐
    - 비 오는 날은 step이 아니라 수 시간에 걸쳐 부드럽게 습도 ↑ / 기온 ↓
    """
    cfg = market.env_cfg
    hours = ax.hours
    n = len(hours)

    # 1) 일교차 (diurnal) — 기온/습도 반비례 부분 (물리 기본)
    temp_diurnal = cfg["temp_c_amp"] * np.sin(2 * np.pi * (hours - 9) / 24.0)
    hum_diurnal = -cfg["humidity_amp"] * 0.6 * np.sin(2 * np.pi * (hours - 9) / 24.0)

    # 2) 슬로우 드리프트 — 며칠 스케일 weather front (두 변수 독립 → 거울 깨짐)
    temp_drift = _slow_drift_centered(n, sigma_frac=0.25, rng=rng) * cfg["temp_c_amp"] * 0.5
    hum_drift = _slow_drift_centered(n, sigma_frac=0.35, rng=rng) * cfg["humidity_amp"] * 0.7

    temp = cfg["temp_c_mean"] + temp_diurnal + temp_drift
    hum = cfg["humidity_mean"] + hum_diurnal + hum_drift

    # 3) 비 에피소드 — 6시간 평활화로 부드러운 ramp up/down
    rain_days = set(cfg.get("rain_days", []))
    if rain_days:
        dom = ax.day_of_month
        rain_mask = np.isin(dom, list(rain_days)).astype(np.float32)
        smooth_k = min(n, 3600 * 6)
        if smooth_k > 1:
            kernel = np.ones(smooth_k, dtype=np.float32) / smooth_k
            rain_smooth = np.convolve(rain_mask, kernel, mode="same")
        else:
            rain_smooth = rain_mask
        hum = hum + rain_smooth * 12.0
        temp = temp - rain_smooth * 2.0

    # 4) 약한 노이즈 (주기성 가독성 유지)
    temp = temp + rng.normal(0, 0.2, size=n)
    hum = hum + rng.normal(0, 0.8, size=n)

    # 5) 소프트 클립 (saturation 회피 — 35~92% 범위)
    hum = np.clip(hum, 35.0, 92.0)

    return {"T_env": temp.astype(np.float32), "H_env": hum.astype(np.float32)}


# ---------- helpers ----------

def _open_mask(profile, hours: np.ndarray, dow: np.ndarray) -> np.ndarray:
    """Smooth open/close via logistic ramps around open/close hour."""
    open_h = profile.daily_open_hour
    close_h = profile.daily_close_hour
    # 30-min ramp width
    k = 6.0
    ramp_up = 1.0 / (1.0 + np.exp(-k * (hours - open_h)))
    ramp_dn = 1.0 / (1.0 + np.exp(k * (hours - close_h)))
    return ramp_up * ramp_dn  # 0..1, ~1 in business hours


def _ar1_noise(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    # AR(1): y[i] = phi*y[i-1] + eps[i], y[0] = eps[0] (lfilter zero ICs).
    eps = rng.normal(0, sigma, size=n).astype(np.float32)
    return lfilter([1.0], [1.0, -phi], eps).astype(np.float32)


def _intraday_modulation(
    hours: np.ndarray,
    morning_amp: float = 0.55,
    lunch_amp: float = 1.0,
    dinner_amp: float = 1.15,
    morning_h: float = 8.5,
    lunch_h: float = 12.5,
    dinner_h: float = 18.5,
) -> np.ndarray:
    """전통시장 일중 활동 패턴 — 아침 입고/오픈러시(~8.5h), 점심(~12.5h), 저녁(~18.5h).
    각 피크 가우시안 합. amplitude는 영업 중 base 위에 더해지는 활동량 가중치."""
    morning = np.exp(-0.5 * ((hours - morning_h) / 1.0) ** 2)
    lunch = np.exp(-0.5 * ((hours - lunch_h) / 1.1) ** 2)
    dinner = np.exp(-0.5 * ((hours - dinner_h) / 1.3) ** 2)
    return (morning_amp * morning + lunch_amp * lunch + dinner_amp * dinner).astype(np.float32)


def _micro_bursts(n: int, rate_per_hour: float, mean_dur_s: float,
                  amp_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Poisson-ish customer bursts. Returns multiplicative factor centered at 1.0."""
    factor = np.ones(n, dtype=np.float32)
    expected = int(rate_per_hour * n / 3600.0)
    if expected <= 0:
        return factor
    n_bursts = max(1, int(rng.poisson(expected)))
    starts = rng.integers(0, n, size=n_bursts)
    durs = np.maximum(1, rng.exponential(mean_dur_s, size=n_bursts).astype(int))
    amps = rng.uniform(0.3, 1.0, size=n_bursts) * amp_frac
    for s, d, a in zip(starts, durs, amps):
        e = min(n, s + d)
        ramp = np.hanning(max(2, e - s)).astype(np.float32)
        factor[s:e] += a * ramp[: e - s]
    return factor


def discretize_q(q: np.ndarray, block_s: int = 15, step: float = 0.5) -> np.ndarray:
    """펄스 카운터 스타일: block_s 윈도우 평균 후 step L/min 스텝으로 양자화 → 계단형 신호.
    실제 디지털 유량계가 pulses/window를 누적해 표시하는 방식 모사."""
    n = len(q)
    pad = (block_s - n % block_s) % block_s
    if pad:
        q_padded = np.concatenate([q, np.zeros(pad, dtype=q.dtype)])
    else:
        q_padded = q
    block_means = q_padded.reshape(-1, block_s).mean(axis=1)
    block_steps = np.round(block_means / step) * step
    return np.repeat(block_steps, block_s)[:n].astype(np.float32)


def _slow_drift(n: int, sigma_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Slow random walk in [1 - 3σ, 1 + 3σ] band, ~hour-scale."""
    # one perturbation per ~600s, then linear interp
    n_knots = max(8, n // 600)
    knots = rng.normal(0, sigma_frac, size=n_knots).cumsum()
    knots = knots - knots.mean()
    xp = np.linspace(0, n - 1, n_knots)
    return (1.0 + np.interp(np.arange(n), xp, knots)).astype(np.float32)


# ---------- per-store channel synth ----------

def store_baseline(
    store: Store,
    ax: TimeAxis,
    env: dict[str, np.ndarray],
    market: Market,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Return dict channel -> 1d float32 array of length ax.n_seconds."""
    n = ax.n_seconds
    hours = ax.hours
    dow = ax.day_of_week
    month = ax.month
    p = store.profile

    open_w = _open_mask(p, hours, dow).astype(np.float32)
    is_weekend = (dow >= 5).astype(np.float32)
    weekend_mult_load = 1.0 + (market.weekly_cfg["weekend_load_mult"] - 1.0) * is_weekend
    weekend_mult_water = 1.0 + (market.weekly_cfg["weekend_water_mult"] - 1.0) * is_weekend

    seasonal_mult = np.ones(n, dtype=np.float32)
    if p.seasonal_boost_months:
        in_season = np.isin(month, p.seasonal_boost_months)
        seasonal_mult = np.where(in_season, 1.5, 1.0).astype(np.float32)

    # intraday morning/lunch/dinner peaks (영업 중에만)
    intraday = _intraday_modulation(hours) * open_w
    # slow drift (~hour scale) — non-stationary normal
    drift_load = _slow_drift(n, sigma_frac=0.04, rng=rng)
    drift_leak = _slow_drift(n, sigma_frac=0.06, rng=rng)
    drift_q = _slow_drift(n, sigma_frac=0.05, rng=rng)
    # customer micro-bursts (short, 10s~3min) — rate scales with how busy the category is
    busy_rate = max(0.5, p.load_kw_day / 2.0)  # bursts/hour during open
    load_bursts = _micro_bursts(n, rate_per_hour=busy_rate * 4, mean_dur_s=45.0,
                                amp_frac=0.5, rng=rng) * (0.3 + 0.7 * open_w)
    water_bursts = _micro_bursts(n, rate_per_hour=busy_rate * 2, mean_dur_s=60.0,
                                 amp_frac=0.8, rng=rng) * (0.2 + 0.8 * open_w)
    # leak follows a similar but slower burst pattern (humidity events)
    leak_bursts = _micro_bursts(n, rate_per_hour=1.5, mean_dur_s=180.0,
                                amp_frac=0.35, rng=rng)

    # ---- I_load (A) ----
    # 영업 외엔 야간 base (냉장/조명 stand-by), 영업 중엔 day base + intraday 피크(morning/lunch/dinner)
    activity = open_w + intraday * 0.7  # intraday 비중 ↑ — 피크 가시성
    kw = p.load_kw_night + (p.load_kw_day - p.load_kw_night) * activity
    kw *= weekend_mult_load * seasonal_mult * drift_load
    i_load = kw * 4.5 * load_bursts  # A @ 220V single-phase (rough)
    i_load = i_load * (1.0 + _ar1_noise(n, market.noise_cfg["ar1_phi"],
                                        market.noise_cfg["ar1_sigma_frac"], rng))
    i_load = np.maximum(i_load, 0.05).astype(np.float32)

    # ---- I_leak (mA) ----
    # 실제 ZCT 측정 기반: 정상 분포는 sub-mA ~ 수 mA. 구성:
    #  (1) capacitive baseline — 회로 항상 존재 (~0.3~0.8 mA, 부하 무관)
    #  (2) humidity-driven leakage — 비선형 (습도 70%+ 부터 가속)
    #  (3) load-coupled leakage — 부하 켜질 때 절연 응력 → 약한 가산
    #  (4) AR(1) noise + 부하 switching 시 짧은 inrush bump
    is_wet = p.leak_threshold_mA <= 15
    cap_baseline = 0.3 if is_wet else 0.15        # mA, 콘덴서/필터 leakage
    leak_base_hum = market.leak_baseline_cfg["wet" if is_wet else "dry"]
    # 비선형 습도 반응 (>70% 부터 가속)
    hum_norm = np.clip((env["H_env"] - 70.0) / 30.0, 0, 1).astype(np.float32)
    hum_term = leak_base_hum * (hum_norm ** 1.8) * 1.4
    # 부하 동작 시 절연 응력 — 매우 약한 비례 (mA per A, 정상은 0.05)
    load_coupling = 0.05 * i_load
    # switching inrush — 부하가 급변할 때 짧은 leak bump (1초 이상 diff)
    load_diff = np.zeros(n, dtype=np.float32)
    load_diff[1:] = np.maximum(0, i_load[1:] - i_load[:-1])
    switching = load_diff * 0.15
    i_leak = (cap_baseline + hum_term + load_coupling + switching) * drift_leak * leak_bursts
    # AR(1) noise (mA 단위 더하기 — 실제 측정 노이즈 0.05~0.2 mA)
    noise_std = 0.08 if is_wet else 0.04
    i_leak = i_leak + _ar1_noise(n, market.noise_cfg["ar1_phi"], noise_std, rng)
    i_leak = np.maximum(i_leak, 0.02).astype(np.float32)

    # ---- Q (L/min) — 펄스 카운터식 계단형, 정상 baseline은 임계 훨씬 아래 ----
    # MNF(야간 minimum night flow)는 0이 아니지만 매우 낮음 (~0.05 L/min)
    mnf_baseline = 0.03 + 0.04 * (1.0 - (env["H_env"] / 100.0))
    # intraday 가중치 ↓: peak이 너무 튀지 않게 (정상은 임계 well below)
    lpm = mnf_baseline + (p.water_lpm_day - p.water_lpm_night) * (open_w + intraday * 0.4)
    lpm *= weekend_mult_water * seasonal_mult * drift_q
    # 영업 종료 후 청소 — amp 축소
    close_h = p.daily_close_hour
    cleaning = np.exp(-0.5 * ((hours - (close_h + 0.5)) / 0.4) ** 2).astype(np.float32) * 0.4
    lpm = lpm + p.water_lpm_day * cleaning
    # 미세 burst — amp 0.8 → 0.2 (계단형 신호 가독성 우선)
    water_burst_small = _micro_bursts(n, rate_per_hour=busy_rate, mean_dur_s=120.0,
                                       amp_frac=0.2, rng=rng) * (0.5 + 0.5 * open_w)
    lpm = lpm * water_burst_small
    lpm = np.maximum(lpm, 0.0).astype(np.float32)
    # 계단형 양자화 — 15s 블록 + 0.5 L/min 스텝 (AR1 노이즈 제거 → clean staircase)
    q = discretize_q(lpm, block_s=15, step=0.5)


    # ---- P (bar) ----
    # 펌프 by-cycle baseline 4 bar, Q 부하 시 dip + 워터해머 잠재성
    p_bar = 4.0 - 0.18 * (q / max(p.water_lpm_day, 0.5))
    # 펌프 사이클링 — 30s~2min 주기로 ±0.05 bar
    pump_cycle = 0.05 * np.sin(2 * np.pi * np.arange(n) / 60.0).astype(np.float32)
    p_bar = p_bar + pump_cycle
    # 단기 진동 (burst 시)
    p_bar = p_bar - 0.10 * (water_bursts - 1.0)
    p_bar = p_bar + rng.normal(0, 0.04, size=n).astype(np.float32)
    p_bar = np.clip(p_bar, 1.5, 5.0).astype(np.float32)

    return {
        "I_leak": i_leak,
        "I_load": i_load,
        "Q": q,
        "P": p_bar,
        "T_env": env["T_env"],
        "H_env": env["H_env"],
    }
