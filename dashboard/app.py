"""Streamlit dashboard — 분전반(SDP) 중심 누전/누수 모니터링.

레이아웃:
  사이드바: 분전반/점포 선택, 기간/seed, 표시 윈도우
  메인:
    [좌] 토폴로지 트리 (TR→MDP→SDP→점포, 분전반 강조)
    [우] 누전 / 누수 / 환경 3그룹 시계열 (임계 라인 + 이상 음영)
  하단: 이상 인스턴스 표 + 임계 초과 점포 + CSV 다운로드
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.simulator import SimResult, simulate  # noqa: E402
from src.anomaly import AnomalyInstance, apply_anomaly  # noqa: E402
from src.baseline import discretize_q  # noqa: E402
from src.viz import (  # noqa: E402
    CHANNEL_LABEL,
    grouped_timeseries_figure,
    market_map_figure,
    topology_tree_figure,
)


st.set_page_config(page_title="시장 누전·누수 모니터링", layout="wide")

# Custom CSS for Premium UI
st.markdown("""
<style>
/* Main container styling */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
/* Card layouts for containers */
div[data-testid="stExpander"] {
    background-color: #f8f9fa;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.04);
    border: 1px solid #e9ecef;
    margin-bottom: 1.5rem;
}
/* Style headers */
h1, h2, h3 {
    font-family: 'Outfit', 'Inter', sans-serif;
    color: #1d3557;
}
/* Metric card design */
.metric-container {
    background: linear-gradient(135deg, #1d3557 0%, #457b9d 100%);
    color: white;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(29, 53, 87, 0.12);
    margin-bottom: 1rem;
    text-align: center;
    transition: transform 0.2s ease-in-out;
}
.metric-container:hover {
    transform: translateY(-3px);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0.2rem 0;
}
.metric-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    opacity: 0.95;
}
</style>
""", unsafe_allow_html=True)


def clone_sim_result(res: SimResult) -> SimResult:
    return SimResult(
        market=res.market,
        ax=res.ax,
        env={k: v.copy() for k, v in res.env.items()},
        store_series={sid: {ch: arr.copy() for ch, arr in channels.items()} for sid, channels in res.store_series.items()},
        label_mask={sid: arr.copy() for sid, arr in res.label_mask.items()},
        nuisance_mask={sid: arr.copy() for sid, arr in res.nuisance_mask.items()},
        panel_aggregates={pid: {ch: arr.copy() for ch, arr in channels.items()} for pid, channels in res.panel_aggregates.items()},
        panel_label_mask={pid: arr.copy() for pid, arr in res.panel_label_mask.items()},
        panel_nuisance_mask={pid: arr.copy() for pid, arr in res.panel_nuisance_mask.items()},
        dong_aggregates={did: {ch: arr.copy() for ch, arr in channels.items()} for did, channels in res.dong_aggregates.items()},
        dong_label_mask={did: arr.copy() for did, arr in res.dong_label_mask.items()},
        anomalies=list(res.anomalies)
    )


def recalculate_aggregates(result: SimResult) -> None:
    n_seconds = result.ax.n_seconds
    market = result.market
    store_series = result.store_series
    label_mask = result.label_mask
    nuisance_mask = result.nuisance_mask

    # Panel aggregates
    panel_agg = {}
    panel_label = {}
    panel_nui = {}
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
        lab = np.zeros(n_seconds, dtype=bool)
        nui = np.zeros(n_seconds, dtype=bool)
        for sid in sid_list:
            lab |= label_mask[sid]
            nui |= nuisance_mask[sid]
        panel_label[panel.panel_id] = lab
        panel_nui[panel.panel_id] = nui

    # Dong aggregates
    dong_agg = {}
    dong_label = {}
    for d in market.dongs:
        sid_list = [s.store_id for s in d.stores]
        if not sid_list:
            continue
        q_main = np.sum([store_series[sid]["Q"] for sid in sid_list], axis=0)
        p_main = np.mean([store_series[sid]["P"] for sid in sid_list], axis=0)
        i_leak_dong = np.sum(
            [panel_agg[p.panel_id]["I_leak_panel"] for p in d.panels if p.panel_id in panel_agg],
            axis=0
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

    result.panel_aggregates.update(panel_agg)
    result.panel_label_mask.update(panel_label)
    result.panel_nuisance_mask.update(panel_nui)
    result.dong_aggregates.update(dong_agg)
    result.dong_label_mask.update(dong_label)


def reapply_all_events(target_result: SimResult, source_baseline: SimResult) -> None:
    # 1. Restore timeseries from baseline
    for sid in target_result.store_series:
        for ch in target_result.store_series[sid]:
            target_result.store_series[sid][ch][:] = source_baseline.store_series[sid][ch]
        target_result.label_mask[sid][:] = source_baseline.label_mask[sid]
        target_result.nuisance_mask[sid][:] = source_baseline.nuisance_mask[sid]
    
    # 2. Re-apply all anomalies
    for a in target_result.anomalies:
        if a.id.startswith("VALVE-"):
            # Random-pattern anomaly on the dong water flow (Q)
            v_dong = a.id.split("-")[1]
            s_idx = a.start_idx
            e_idx = a.end_idx
            pattern = a.extras.get("pattern", "spike")
            dong_obj = next((d for d in target_result.market.dongs if d.dong_id == v_dong), None)
            if dong_obj is None:
                continue
            mag = _PATTERN_MAGNITUDE["Q"][pattern]
            for st_obj in dong_obj.stores:
                _apply_pattern(
                    target_result.store_series[st_obj.store_id]["Q"],
                    s_idx, e_idx, pattern, mag,
                )
                target_result.label_mask[st_obj.store_id][s_idx:e_idx] = True
        elif a.id.startswith("BREAK-"):
            # Random-pattern anomaly on panel electrical (I_load + I_leak)
            b_panel = a.id.split("-")[1]
            s_idx = a.start_idx
            e_idx = a.end_idx
            pattern = a.extras.get("pattern", "spike")
            panel_obj = target_result.market.panel_by_id(b_panel)
            for st_obj in panel_obj.stores:
                _apply_pattern(
                    target_result.store_series[st_obj.store_id]["I_load"],
                    s_idx, e_idx, pattern, _PATTERN_MAGNITUDE["I_load"][pattern],
                )
                _apply_pattern(
                    target_result.store_series[st_obj.store_id]["I_leak"],
                    s_idx, e_idx, pattern, _PATTERN_MAGNITUDE["I_leak"][pattern],
                )
                target_result.label_mask[st_obj.store_id][s_idx:e_idx] = True
        else:
            # Re-apply manual anomaly
            store_obj = target_result.market.store_by_id(a.store_id)
            apply_anomaly(
                a,
                target_result.store_series[a.store_id],
                target_result.env,
                target_result.ax,
                store_obj,
                target_result.label_mask[a.store_id],
                target_result.nuisance_mask[a.store_id]
            )
            
    # 2.5. Q 채널 재이산화 — anomaly가 적용된 후에도 계단형 유지 (펄스 카운터 모사)
    for sid in target_result.store_series:
        q_arr = target_result.store_series[sid]["Q"]
        target_result.store_series[sid]["Q"] = discretize_q(q_arr)

    # 3. Recalculate aggregates
    recalculate_aggregates(target_result)


# ---------- random-pattern hot-trigger helpers ----------

_PATTERN_CHOICES = ["spike", "ctx_inc"]
_PATTERN_LABEL_KR = {
    "spike": "스파이크 ↑",
    "ctx_inc": "점진 증가",
}
# magnitude per channel per pattern — 임계치(점포 15-30mA / 패널 100mA / MDP 300mA)
# 위로 확실히 끌어올리는 크기로 설정. per-store 값.
_PATTERN_MAGNITUDE = {
    "Q":      {"spike": 15.0,  "ctx_inc": 10.0},
    "I_load": {"spike": 25.0,  "ctx_inc": 18.0},
    "I_leak": {"spike": 120.0, "ctx_inc": 80.0},
}


def _apply_pattern(
    arr: np.ndarray, s_idx: int, e_idx: int, pattern: str, magnitude: float
) -> None:
    """Modify arr[s_idx:e_idx] in-place using the named pattern shape."""
    if e_idx <= s_idx:
        return
    dur = e_idx - s_idx
    ts = np.arange(dur, dtype=np.float32)
    region = arr[s_idx:e_idx]
    if pattern == "spike":
        peak = dur // 4
        sigma = max(2.0, dur / 8.0)
        bump = magnitude * np.exp(-((ts - peak) ** 2) / (2 * sigma ** 2))
        arr[s_idx:e_idx] = region + bump.astype(arr.dtype)
    elif pattern == "ctx_inc":
        ramp = (magnitude * (ts / max(1, dur - 1))).astype(arr.dtype)
        arr[s_idx:e_idx] = region + ramp


def _placement_middle(
    press_idx: int, delay_s: int, dur_s: int, n_total: int,
    busy: list[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    """anomaly 윈도우 [s_idx, e_idx]를 타임라인 중간 80% [n*0.1, n*0.9] 안에서
    기존 anomaly 구간(busy)과 겹치지 않게 desired 시점에 가장 가까운 빈 구간에 배치."""
    lo = int(n_total * 0.1)
    hi = int(n_total * 0.9)
    desired = press_idx + delay_s

    busy_sorted = sorted(busy or [])

    # [lo, hi] 안의 빈 구간(gap) 목록
    gaps: list[tuple[int, int]] = []
    cur = lo
    for bs, be in busy_sorted:
        if be < lo or bs > hi:
            continue
        bs_c = max(bs, lo)
        be_c = min(be, hi)
        if cur < bs_c:
            gaps.append((cur, bs_c - 1))
        cur = max(cur, be_c + 1)
    if cur <= hi:
        gaps.append((cur, hi))

    # dur_s가 들어갈 수 있는 gap 중 desired에 가장 가까운 시작점 선택
    candidates = [(gs, ge) for gs, ge in gaps if ge - gs + 1 >= dur_s]
    if candidates:
        best_s = None
        best_d = float("inf")
        for gs, ge in candidates:
            s_try = max(gs, min(ge - dur_s + 1, desired))
            d = abs(s_try - desired)
            if d < best_d:
                best_d = d
                best_s = s_try
        s_idx = best_s
    else:
        # 빈자리가 없으면 fallback — 중앙 클램프 (겹침 허용)
        if hi - lo < dur_s:
            s_idx = max(0, n_total // 2 - dur_s // 2)
        else:
            s_idx = max(lo, min(hi - dur_s, desired))

    e_idx = min(n_total - 1, s_idx + dur_s)
    return s_idx, e_idx


def resample_series_dict(series_dict: dict, freq: str, agg: str) -> dict:
    res = {}
    for k, v in series_dict.items():
        resampler = v.resample(freq)
        res[k] = getattr(resampler, agg)()
    return res



# ---------- cache ----------

def _config_signature() -> tuple[tuple[str, float], ...]:
    """mtime of each config YAML — invalidates cache when configs change."""
    cfg_dir = ROOT / "config"
    return tuple(sorted(
        (p.name, p.stat().st_mtime) for p in cfg_dir.glob("*.yaml")
    ))


@st.cache_resource(show_spinner=True)
def _run_sim(days: int, seed: int, config_sig: tuple) -> SimResult:
    return simulate(ROOT / "config", days=days, seed=seed)


@st.cache_data
def _topology_store_count(config_sig: tuple) -> int:
    with open(ROOT / "config" / "market_topology.yaml", encoding="utf-8") as f:
        topo = yaml.safe_load(f)
    return sum(int(d["n_stores"]) for d in topo["dongs"])


# ---------- status helpers ----------

def _store_status_in_window(result: SimResult, t0: int, t1: int) -> dict[str, str]:
    """Mark store status by anomalies active inside [t0, t1]."""
    st_map: dict[str, str] = {}
    for a in result.anomalies:
        if a.end_idx < t0 or a.start_idx > t1:
            continue
        is_nui = a.scenario_type == "N1_inverter_falsetrip"
        cur = st_map.get(a.store_id)
        if is_nui and cur != "alert":
            st_map[a.store_id] = "nuisance"
        else:
            st_map[a.store_id] = "alert"
    return st_map


def _panel_status_from_stores(result: SimResult, store_status: dict[str, str]) -> dict[str, str]:
    panel_status: dict[str, str] = {}
    for panel in result.market.panels:
        statuses = [store_status.get(s.store_id) for s in panel.stores]
        if any(x == "alert" for x in statuses):
            panel_status[panel.panel_id] = "alert"
        elif any(x == "nuisance" for x in statuses):
            panel_status[panel.panel_id] = "nuisance"
    return panel_status


def _first_crossing(arr: np.ndarray, s_idx: int, e_idx: int, threshold: float) -> int | None:
    """[s_idx, e_idx] 범위에서 arr이 threshold를 처음 넘는 인덱스. 없으면 None."""
    if e_idx <= s_idx or s_idx >= len(arr):
        return None
    e_clip = min(e_idx + 1, len(arr))
    region = arr[s_idx:e_clip]
    above = np.where(region >= threshold)[0]
    if len(above) == 0:
        return None
    return s_idx + int(above[0])


def _anomaly_spans_for_stores(
    result: SimResult, store_ids: set[str], t0: int, t1: int,
    panel_id: str | None = None, dong_id: str | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp, str, str]]:
    """음영 시작점 = 신호가 실제로 해당 임계를 처음 넘는 시점 (domain knowledge).
    crossing이 없으면 a.start_idx로 fallback."""
    idx = result.ax.index
    spans = []
    color_map = {"N1_inverter_falsetrip": "rgba(244,162,97,0.5)"}
    default = "rgba(230,57,70,0.4)"

    # 임계치 사전 계산
    panel_obj = result.market.panel_by_id(panel_id) if panel_id else None
    dong_obj = next((d for d in result.market.dongs if d.dong_id == dong_id), None) if dong_id else None
    panel_thr = panel_obj.trip_threshold_mA if panel_obj else None
    mnf_thr = (dong_obj.n_stores * 8.0) if dong_obj else None  # 차트 누수 임계와 동일

    for a in result.anomalies:
        # 매칭: (1) 실제 점포 ID 일치, 또는 (2) VALVE-{dong}/BREAK-{panel}이 선택과 일치
        is_valve = a.id.startswith("VALVE-")
        is_break = a.id.startswith("BREAK-")
        matches = a.store_id in store_ids
        if not matches and is_valve and dong_id is not None:
            matches = a.id.split("-")[1] == dong_id
        if not matches and is_break and panel_id is not None:
            matches = a.id.split("-")[1] == panel_id
        if not matches:
            continue
        if a.end_idx < t0 or a.start_idx > t1:
            continue

        # 음영 시작 = 임계 crossing 시점
        s_pos = a.start_idx
        if is_valve and dong_obj is not None and mnf_thr is not None:
            q_main = result.dong_aggregates[dong_obj.dong_id]["Q_main"]
            crossing = _first_crossing(q_main, a.start_idx, a.end_idx, mnf_thr)
            if crossing is not None:
                s_pos = crossing
        elif is_break and panel_obj is not None and panel_thr is not None:
            i_leak_p = result.panel_aggregates[panel_obj.panel_id]["I_leak_panel"]
            crossing = _first_crossing(i_leak_p, a.start_idx, a.end_idx, panel_thr)
            if crossing is not None:
                s_pos = crossing

        s = idx[max(s_pos, t0)]
        e = idx[min(a.end_idx, t1 - 1)]
        color = color_map.get(a.scenario_type, default)
        spans.append((s, e, color, f"{a.id} · {a.scenario_type} · {a.channel}"))
    return spans


# ---------- sidebar ----------

st.sidebar.title("⚙️ 시뮬레이터")
days = st.sidebar.slider("기간 (일)", 1, 14, 7,
                        help="14일 이상이 필요하면 CLI 사용 권장 — `python -m src.simulator --days N`")
# 비용 추정 (점포 × 채널 × 초 × 4byte)
_n_stores_cfg = _topology_store_count(_config_signature())
_est_gb = days * 86400 * _n_stores_cfg * 6 * 4 / (1024 ** 3)
_est_sec = max(2, int(days * _n_stores_cfg / 25.0))
st.sidebar.caption(
    f"📦 추정: 점포 {_n_stores_cfg}곳 · 메모리 ~{_est_gb:.1f} GB · 소요 ~{_est_sec}초"
)
if days >= 10:
    st.sidebar.warning(f"⚠️ {days}일은 무거움. 슬라이더는 변경해도 자동 재실행 안 함 — "
                       "버튼으로만 시작합니다.")
seed = st.sidebar.number_input("seed", value=42, step=1)
run_btn = st.sidebar.button("▶ 시뮬레이션 실행", type="primary")

if "result" not in st.session_state:
    st.session_state.result = None

# 명시적 버튼만 실행 트리거. 슬라이더 조정으로 자동 재실행 안 함.
if run_btn:
    with st.spinner(f"{days}일 시뮬레이션 — 진행 로그는 터미널 참고"):
        raw_res = _run_sim(days, int(seed), _config_signature())
        st.session_state.result = clone_sim_result(raw_res)

if st.session_state.result is None:
    st.info("👈 좌측 사이드바에서 **기간/seed**를 정하고 **▶ 시뮬레이션 실행** 버튼을 누르세요. "
            "(자동 실행 없음 — 슬라이더만 옮겨도 새로 돌지 않습니다.)")
    st.stop()

result: SimResult = st.session_state.result
n = result.ax.n_seconds
idx = result.ax.index
# 현재 result가 어떤 설정으로 만들어졌는지 표시
_cur_days = result.ax.n_seconds // 86400
st.sidebar.success(f"✅ 현재 결과: **{_cur_days}일** (seed={seed})")
if _cur_days != days:
    st.sidebar.info(f"슬라이더는 {days}일이지만 결과는 {_cur_days}일치예요. "
                    "버튼을 다시 누르면 재시뮬레이션.")

st.sidebar.markdown("---")

# 분전반 선택 (default = 점포가 가장 많은 분전반)
panel_options = [p.panel_id for p in result.market.panels]
panel_labels = {p.panel_id: f"{p.panel_id} · {p.name} ({len(p.stores)}점포)"
                for p in result.market.panels}

# Hot-trigger 버튼이 요청한 패널/점포로 자동 전환
if "pending_pid" in st.session_state:
    target_pid = st.session_state.pop("pending_pid")
    if target_pid in panel_options:
        st.session_state["sb_pid"] = target_pid
        st.session_state.pop("sb_sid", None)

pid = st.sidebar.selectbox(
    "🔌 분전반(SDP)", panel_options,
    format_func=lambda x: panel_labels[x],
    key="sb_pid",
)
panel = result.market.panel_by_id(pid)

# 분전반 소속 점포 중 선택
store_options = [s.store_id for s in panel.stores]
if st.session_state.get("sb_sid") not in store_options:
    st.session_state.pop("sb_sid", None)
sid = st.sidebar.selectbox(
    "🏪 점포 (분전반 내)", store_options,
    key="sb_sid",
)
store = result.market.store_by_id(sid)

# 시간
t_hours = st.sidebar.slider("스냅샷 시각 (시간)", 0, max(1, n // 3600 - 1), 12)
t_idx = min(n - 1, t_hours * 3600)
window_h = st.sidebar.slider("표시 윈도우 (시간)", 1, 168, 48)

t0 = max(0, t_idx - window_h * 1800)
t1 = min(n, t_idx + window_h * 1800)

# 차트 시간 해상도 및 집계 방식
st.sidebar.markdown("---")
st.sidebar.subheader("📊 차트 보기 설정")
time_res_opt = st.sidebar.selectbox(
    "시간 해상도 (Interval)",
    ["1분 (1m)", "30분 (30m)", "1시간 (1H)"],
    index=0,
)
agg_method_opt = st.sidebar.selectbox(
    "차트 집계 방식 (Aggregation)",
    ["평균 (Mean)", "최대값 (Max)", "최소값 (Min)", "합계 (Sum)", "마지막값 (Last)"],
    index=0,
)

time_res_map = {
    "1분 (1m)": "1min",
    "30분 (30m)": "30min",
    "1시간 (1H)": "1h",
}
agg_method_map = {
    "평균 (Mean)": "mean",
    "최대값 (Max)": "max",
    "최소값 (Min)": "min",
    "합계 (Sum)": "sum",
    "마지막값 (Last)": "last",
}

time_res_str = time_res_map[time_res_opt]
agg_method_str = agg_method_map[agg_method_opt]



# ---------- top: topology + summary ----------

st.title("⚡💧 전통시장 누전·누수 모니터링")
_n_panels = sum(1 for _ in result.market.panels)
_n_stores = sum(1 for _ in result.market.stores)
st.caption(f"시뮬레이션 기간: {idx[0]} ~ {idx[-1]} · {days}일 · "
           f"분전반 {_n_panels}개 · 점포 {_n_stores}곳")

store_status = _store_status_in_window(result, t0, t1)
panel_status = _panel_status_from_stores(result, store_status)
panel_store_ids = {s.store_id for s in panel.stores}
spans_leak = _anomaly_spans_for_stores(
    result, panel_store_ids, t0, t1,
    panel_id=pid, dong_id=panel.dong_id,
)

# KPI Metrics Dashboard
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    alert_cnt = sum(1 for status in store_status.values() if status == 'alert')
    st.markdown(f"""
    <div class="metric-container" style="background: linear-gradient(135deg, #e63946 0%, #9d0208 100%);">
        <div class="metric-label">🚨 위험 경보 점포</div>
        <div class="metric-value">{alert_cnt} / {_n_stores}</div>
        <div class="metric-label">이상 감지 점포 수</div>
    </div>
    """, unsafe_allow_html=True)
with kpi2:
    max_leak_val = max(float(result.store_series[s.store_id]["I_leak"].max()) for s in result.market.stores)
    st.markdown(f"""
    <div class="metric-container" style="background: linear-gradient(135deg, #f4a261 0%, #e76f51 100%);">
        <div class="metric-label">🔌 최대 누설 전류</div>
        <div class="metric-value">{max_leak_val:.1f} mA</div>
        <div class="metric-label">점포 최대 누설 기록</div>
    </div>
    """, unsafe_allow_html=True)
with kpi3:
    nui_cnt = sum(1 for status in store_status.values() if status == 'nuisance')
    st.markdown(f"""
    <div class="metric-container" style="background: linear-gradient(135deg, #457b9d 0%, #1d3557 100%);">
        <div class="metric-label">🟠 오동작(Nuisance) 점포</div>
        <div class="metric-value">{nui_cnt} 곳</div>
        <div class="metric-label">ELCB 오동작 경보</div>
    </div>
    """, unsafe_allow_html=True)

# ===== 상단: 토폴로지 풀폭 (지도 / 트리 토글) =====
hdr_l, hdr_r = st.columns([3, 1])
with hdr_l:
    st.subheader("🗺️ 시장 토폴로지")
with hdr_r:
    view_mode = st.radio(
        "보기", ["🏪 시장 지도", "🌳 계통 트리"], horizontal=True,
        label_visibility="collapsed",
    )

if view_mode == "🏪 시장 지도":
    st.plotly_chart(
        market_map_figure(result.market,
                          panel_status=panel_status,
                          store_status=store_status,
                          highlight_panel=pid),
        use_container_width=True,
    )
else:
    st.plotly_chart(
        topology_tree_figure(result.market,
                             panel_status=panel_status,
                             store_status=store_status,
                             highlight_panel=pid),
        use_container_width=True,
    )
st.caption(
    "🟢 정상 · 🔴 이상 · 🟠 nuisance(N1) — "
    f"선택 분전반 **{pid}** 노란 테두리 강조 · "
    f"창 내 이상 점포 {len(store_status)}곳 · "
    f"표시 윈도우 내 이상 인스턴스 {len(spans_leak)}건"
)

# ===== 좌측: 핫트리거 / 우측: 시계열 3종 =====
baseline_res = _run_sim(_cur_days, int(seed), _config_signature())
trig_col, chart_col = st.columns([1, 3], gap="medium")

# ---------- 좌측: 원클릭 핫트리거 ----------
with trig_col:
    st.subheader("⚡ 계통망 원클릭 핫트리거")
    st.caption("동별 수도 밸브 차단 / 분전반 차단기 트립을 한 번 클릭으로 주입 (스냅샷 시각 기준).")

    # Delay & duration
    delay_min = st.slider(
        "⏱️ 발생 지연 (분)", 1, 30, 5,
        help="버튼을 누른 시점 기준으로 몇 분 뒤에 anomaly가 시작되는지",
    )
    dur_min = st.slider("⏳ 지속 시간 (분)", 5, 240, 60)
    delay_s = int(delay_min * 60)
    dur_s = int(dur_min * 60)

    # 동 그리드 — 좁은 폭에 맞춰 2열
    ht_cols = st.columns(2)
    for d_idx, d in enumerate(result.market.dongs):
        with ht_cols[d_idx % 2]:
            st.markdown(f"**📍 {d.name} ({d.dong_id})**")

            # ----- Valve random-pattern trigger -----
            valve_anoms = [a for a in result.anomalies if a.id.startswith(f"VALVE-{d.dong_id}-")]
            if valve_anoms:
                last = valve_anoms[-1]
                pat = _PATTERN_LABEL_KR.get(last.extras.get("pattern", ""), "—")
                st.caption(f"활성 **{len(valve_anoms)}건** · 최근 패턴 **{pat}**")
                if st.button(f"💧 {d.dong_id} 복구", key=f"ht_valve_{d.dong_id}",
                             type="primary", use_container_width=True):
                    result.anomalies = [a for a in result.anomalies
                                        if not a.id.startswith(f"VALVE-{d.dong_id}-")]
                    reapply_all_events(result, baseline_res)
                    st.success(f"{d.name} 밸브 anomaly 제거.")
                    st.rerun()
            else:
                if st.button(f"🔒 {d.dong_id} 밸브 차단", key=f"ht_valve_{d.dong_id}",
                             type="secondary", use_container_width=True):
                    press_idx = int(t_hours) * 3600
                    n_inj = random.randint(2, 4)
                    patterns_used = []
                    for i in range(n_inj):
                        if i == 0:
                            cur_press = press_idx
                        else:
                            # 나머지는 중간 구간에서 랜덤하게 분산
                            cur_press = random.randint(int(n * 0.1), int(n * 0.85)) - delay_s
                        busy = [(a.start_idx, a.end_idx) for a in result.anomalies]
                        s_idx, e_idx = _placement_middle(cur_press, delay_s, dur_s, n, busy=busy)
                        pattern = random.choice(_PATTERN_CHOICES)
                        patterns_used.append(_PATTERN_LABEL_KR[pattern])
                        ins_v = AnomalyInstance(
                            id=f"VALVE-{d.dong_id}-{len(result.anomalies) + 1:03d}",
                            store_id=f"관로 ({d.dong_id})",
                            channel="Q",
                            scenario_type="밸브차단",
                            start_idx=s_idx,
                            end_idx=e_idx,
                            severity=1.0,
                            label_kind="range",
                            extras={"pattern": pattern, "press_idx": cur_press, "delay_s": delay_s},
                        )
                        result.anomalies.append(ins_v)
                    reapply_all_events(result, baseline_res)
                    if d.panels:
                        st.session_state["pending_pid"] = d.panels[0].panel_id
                    st.success(
                        f"🔒 {d.dong_id} 차단 → **{n_inj}건** 주입 "
                        f"({', '.join(patterns_used)})"
                    )
                    st.rerun()

            # ----- SDP breaker random-pattern triggers -----
            for p in d.panels:
                break_anoms = [a for a in result.anomalies if a.id.startswith(f"BREAK-{p.panel_id}-")]
                if break_anoms:
                    last = break_anoms[-1]
                    pat = _PATTERN_LABEL_KR.get(last.extras.get("pattern", ""), "—")
                    st.caption(f"{p.panel_id} · **{len(break_anoms)}건** · 최근 **{pat}**")
                    if st.button(f"🔌 {p.panel_id} 복구", key=f"ht_break_{p.panel_id}",
                                 type="primary", use_container_width=True):
                        result.anomalies = [a for a in result.anomalies
                                            if not a.id.startswith(f"BREAK-{p.panel_id}-")]
                        reapply_all_events(result, baseline_res)
                        st.success(f"{p.panel_id} 차단기 anomaly 제거.")
                        st.rerun()
                else:
                    if st.button(f"⚡ {p.panel_id} 트립", key=f"ht_break_{p.panel_id}",
                                 type="secondary", use_container_width=True):
                        press_idx = int(t_hours) * 3600
                        n_inj = random.randint(2, 4)
                        patterns_used = []
                        for i in range(n_inj):
                            if i == 0:
                                cur_press = press_idx
                            else:
                                cur_press = random.randint(int(n * 0.1), int(n * 0.85)) - delay_s
                            busy = [(a.start_idx, a.end_idx) for a in result.anomalies]
                            s_idx, e_idx = _placement_middle(cur_press, delay_s, dur_s, n, busy=busy)
                            pattern = random.choice(_PATTERN_CHOICES)
                            patterns_used.append(_PATTERN_LABEL_KR[pattern])
                            ins_b = AnomalyInstance(
                                id=f"BREAK-{p.panel_id}-{len(result.anomalies) + 1:03d}",
                                store_id=f"분전반 ({p.panel_id})",
                                channel="I_load",
                                scenario_type="차단기트립",
                                start_idx=s_idx,
                                end_idx=e_idx,
                                severity=1.0,
                                label_kind="range",
                                extras={"pattern": pattern, "press_idx": cur_press, "delay_s": delay_s},
                            )
                            result.anomalies.append(ins_b)
                        reapply_all_events(result, baseline_res)
                        st.session_state["pending_pid"] = p.panel_id
                        st.success(
                            f"⚡ {p.panel_id} 트립 → **{n_inj}건** 주입 "
                            f"({', '.join(patterns_used)})"
                        )
                        st.rerun()

    st.markdown("---")
    if st.button("🔄 모든 핫트리거 초기화", type="secondary", use_container_width=True):
        raw_res = _run_sim(_cur_days, int(seed), _config_signature())
        st.session_state.result = clone_sim_result(raw_res)
        st.success("모든 수동 trigger 초기화 완료.")
        st.rerun()

# ---------- 우측: 시계열 3종 ----------
with chart_col:
    # ===== 누전 =====
    st.subheader("🔌 누전 (Leakage Current)")
    leak_series = {
        "I_leak_panel": pd.Series(result.panel_aggregates[pid]["I_leak_panel"][t0:t1],
                                  index=idx[t0:t1]),
        "I_leak": pd.Series(result.store_series[sid]["I_leak"][t0:t1], index=idx[t0:t1]),
        "I_leak_dong": pd.Series(result.dong_aggregates[panel.dong_id]["I_leak_dong"][t0:t1],
                                 index=idx[t0:t1]),
    }
    thresholds_leak = {
        "I_leak": [(store.profile.leak_threshold_mA, "#e63946",
                    f"점포 임계 {store.profile.leak_threshold_mA:.0f}mA")],
        "I_leak_panel": [(panel.trip_threshold_mA, "#9d0208",
                          f"SDP trip {panel.trip_threshold_mA:.0f}mA")],
        "I_leak_dong": [(result.market.mdp_trip_threshold_mA, "#6a040f",
                         f"MDP trip {result.market.mdp_trip_threshold_mA:.0f}mA")],
    }
    leak_series_ds = resample_series_dict(leak_series, time_res_str, agg_method_str)
    st.plotly_chart(
        grouped_timeseries_figure(
            leak_series_ds, thresholds=thresholds_leak,
            anomaly_spans=spans_leak,
            title=f"{pid} 분전반 누설전류 + 점포 {sid}  ·  {time_res_opt} {agg_method_opt}",
            height=340,
        ),
        use_container_width=True,
    )

    # ===== 누수 =====
    st.subheader("💧 누수 (Water Flow & Pressure)")
    water_series = {
        "Q_main": pd.Series(result.dong_aggregates[panel.dong_id]["Q_main"][t0:t1],
                            index=idx[t0:t1]),
        "Q": pd.Series(result.store_series[sid]["Q"][t0:t1], index=idx[t0:t1]),
        "P_main": pd.Series(result.dong_aggregates[panel.dong_id]["P_main"][t0:t1],
                            index=idx[t0:t1]),
    }
    # 누수 임계: daytime 정상 활동(점심/저녁 피크 + 청소버스트) 훨씬 위에 위치.
    # 정상 baseline은 점포 수 × 2~3 L/min 수준 → 임계 n_stores × 8 로 3~4x 마진 확보
    _dong_obj = next((d for d in result.market.dongs if d.dong_id == panel.dong_id), None)
    _n_stores_in_dong = _dong_obj.n_stores if _dong_obj else 1
    mnf_thr = _n_stores_in_dong * 8.0
    thresholds_water = {
        "Q_main": [(mnf_thr, "#1d3557", f"누수 임계 {mnf_thr:.0f} L/min")],
        "P_main": [(2.0, "#9d0208", "저압 2.0 bar")],
    }
    water_series_ds = resample_series_dict(water_series, time_res_str, agg_method_str)
    st.plotly_chart(
        grouped_timeseries_figure(
            water_series_ds, thresholds=thresholds_water,
            anomaly_spans=spans_leak,
            title=f"{panel.dong_id} 동 메인 + 점포 {sid}  ·  압력은 우측 축",
            height=340, secondary_y_channels=["P_main"],
        ),
        use_container_width=True,
    )

    # ===== 환경 =====
    st.subheader("🌡️ 환경 (Environment)")
    env_series = {
        "T_env": pd.Series(result.env["T_env"][t0:t1], index=idx[t0:t1]),
        "H_env": pd.Series(result.env["H_env"][t0:t1], index=idx[t0:t1]),
    }
    env_ds = resample_series_dict(env_series, time_res_str, agg_method_str)
    st.plotly_chart(
        grouped_timeseries_figure(
            env_ds, thresholds={"H_env": [(80.0, "#e63946", "고습 위험선 80%")]},
            title=f"기온 / 습도  ·  습도는 우측 축",
            height=280, secondary_y_channels=["H_env"],
        ),
        use_container_width=True,
    )


# ---------- bottom: tables + downloads ----------

st.markdown("---")
c3, c4, c5 = st.columns([1.3, 1.0, 0.9])

with c3:
    st.subheader("📋 이상 인스턴스 (선택 분전반 소속)")
    rows = []
    for a in result.anomalies:
        if a.store_id not in panel_store_ids:
            continue
        rows.append({
            "id": a.id, "store": a.store_id, "channel": a.channel,
            "type": a.scenario_type, "severity": round(a.severity, 2),
            "start": str(idx[a.start_idx]), "end": str(idx[a.end_idx]),
        })
    df_anom = pd.DataFrame(rows)
    st.dataframe(df_anom, height=260, use_container_width=True)
    if rows:
        st.caption("이상 타입 카운트 — "
                   + " · ".join(f"{k}={v}" for k, v in
                                df_anom["type"].value_counts().items()))

with c4:
    st.subheader("⚠️ 임계 초과 점포 (전체)")
    rows = []
    for s in result.market.stores:
        thr = s.profile.leak_threshold_mA
        leak = result.store_series[s.store_id]["I_leak"]
        over_frac = float(np.mean(leak >= thr))
        peak = float(leak.max())
        if over_frac > 0:
            rows.append({"store": s.store_id, "panel": s.panel_id,
                         "cat": s.profile.category, "thr_mA": thr,
                         "peak_mA": round(peak, 2),
                         "over_frac": round(over_frac, 4)})
    df_thr = pd.DataFrame(rows).sort_values("over_frac", ascending=False) \
        if rows else pd.DataFrame()
    st.dataframe(df_thr, height=260, use_container_width=True)

with c5:
    st.subheader("⬇️ 데이터 내보내기")
    from src.exporter import build_wide_market
    df_m, lab_m = build_wide_market(result, 60)
    csv_data = df_m.to_csv(float_format="%.4f").encode("utf-8")
    lab_data = lab_m.to_csv().encode("utf-8")
    st.download_button("market_1m.csv", csv_data,
                       file_name=f"market_1m_d{days}_s{seed}.csv",
                       mime="text/csv")
    st.download_button("market_1m.label.csv", lab_data,
                       file_name=f"market_1m_d{days}_s{seed}.label.csv",
                       mime="text/csv")
    meta = {
        "market": result.market.name,
        "start": str(result.ax.start),
        "n_seconds": n,
        "instances": [a.to_dict() for a in result.anomalies],
    }
    st.download_button(
        "anomaly_index.json",
        json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="anomaly_index.json", mime="application/json",
    )

# 시나리오 설명
with st.expander("📖 시나리오 설명"):
    st.markdown("""
| ID | 패턴 | 채널 | 설명 |
|---|---|---|---|
| **A1_drift** | 점진 상승 | I_leak / Q | 시간당 ε% 누적 — 수일 단위 절연 열화 초기 |
| **A2_spike** | 단발 스파이크 | 4채널 | 1~5분 가우시안 펄스 |
| **A3_intermittent** | 야간 + 고습 ON | I_leak | 새벽 0~4시 + 습도 80%+ 조건부 |
| **A4_insulation_degradation** | drift + step | I_leak | linear ramp + 계단형 점프 (12h~72h) |
| **A5_water_hammer** | 감쇠 진동 | P | 밸브 급폐 시 압력 진동 (수 초) |
| **A6_night_leak** | 야간 baseline ↑ | Q | 23시~05시 점진 누수 (180~480분) |
| **A7_load_leak_coupled** | 부하 비례 누설 | I_leak | I_load × severity, 부하 동작 시만 |
| **N1_inverter_falsetrip** | inrush + ELCB trip | I_load,I_leak | 정상 inverter inrush가 ELCB 오동작 → label=nuisance |

빨간 음영 = A류 이상 / 주황 음영 = N1 nuisance.
""")
