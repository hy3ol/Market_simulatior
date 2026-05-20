"""Visualization helpers — hierarchical tree topology + grouped time-series plots.

채널 그룹:
  - 누전(Leakage): I_leak (점포), I_leak_panel (분전반), I_leak_dong (동)
  - 누수(Water):  Q (점포/동), P (점포/동)
  - 환경(Env):    T_env, H_env
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .topology import Market


# 한국어 라벨 + 단위
CHANNEL_LABEL = {
    "I_leak": "점포 누설전류 (mA)",
    "I_load": "점포 부하전류 (A)",
    "Q": "점포 유량 (L/min)",
    "P": "점포 압력 (bar)",
    "T_env": "기온 (°C)",
    "H_env": "습도 (%RH)",
    "I_leak_panel": "분전반 누설전류 합 (mA)",
    "I_load_panel": "분전반 부하전류 합 (A)",
    "I_leak_dong": "동(MDP측) 누설전류 합 (mA)",
    "Q_main": "동 메인 유량 합 (L/min)",
    "P_main": "동 메인 평균 압력 (bar)",
}

GROUP_LEAK = ["I_leak_panel", "I_leak", "I_leak_dong"]
GROUP_WATER = ["Q_main", "Q", "P_main", "P"]
GROUP_ENV = ["T_env", "H_env"]


# ---------- hierarchical tree ----------

def topology_tree_figure(
    market: Market,
    panel_status: dict[str, str] | None = None,
    store_status: dict[str, str] | None = None,
    highlight_panel: str | None = None,
) -> go.Figure:
    """4-tier hierarchical tree: TR → MDP → SDP → store (각 SDP 아래 격자)."""
    g = market.graph
    levels = _compute_levels(g, market)
    pos, layout_info = _hierarchical_layout(g, market, levels)
    max_rows = layout_info["max_store_rows"]
    row_dy = layout_info["row_dy"]
    water_y = layout_info["water_y"]
    x_left = layout_info["x_left"]
    x_right = layout_info["x_right"]

    color_map = {"normal": "#3aa55d", "alert": "#e63946", "nuisance": "#f4a261"}

    # 0) 동별 큰 박스 — SDP 구역부터 점포 격자까지 묶음
    # 수도 헥사곤은 박스 밖(아래)에서 짧은 연결선으로 들어옴
    shapes: list[dict] = []
    dong_box_y_top = -1.10
    dong_box_y_bot = water_y + 0.30   # 수도 헥사곤보다 위 (헥사곤 박스 밖)
    for d in market.dongs:
        if not d.panels:
            continue
        panel_xs = [pos[p.panel_id][0] for p in d.panels if p.panel_id in pos]
        if not panel_xs:
            continue
        xmin = min(panel_xs) - 0.07
        xmax = max(panel_xs) + 0.07
        fill_l = _light_fill(
            _DONG_FILL.get(d.dong_id, "rgba(200,200,200,0.55)"), alpha=0.13
        )
        border = _DONG_BORDER.get(d.dong_id, "#666")
        shapes.append(dict(
            type="rect", x0=xmin, y0=dong_box_y_bot, x1=xmax, y1=dong_box_y_top,
            fillcolor=fill_l,
            line=dict(color=border, width=2),
            layer="below",
        ))

    # 0-b) 수도 인입선 — 각 동의 수도 헥사곤(박스 밖)에서 박스 하단으로 짧은 점선
    water_x_list, water_y_list = [], []
    for d in market.dongs:
        wm = d.main_water_meter_id
        if wm not in pos or not d.panels:
            continue
        wm_x, wm_y = pos[wm]
        # hex 에서 박스 바닥까지 수직 연결
        water_x_list.extend([wm_x, wm_x, None])
        water_y_list.extend([wm_y, dong_box_y_bot, None])

    # 1) edges — water_branch 제외, 소스 종류별 midy 고정
    source_midy = {
        "transformer": -0.25,    # TR(0) ↔ MDP(-0.5) 중간
        "mdp": -1.75,            # 동 박스 안쪽 — 드롭은 동 박스 내부에서 발생
        "panel": -2.42,
    }
    edge_x, edge_y = [], []
    for u, v in g.edges():
        if u not in pos or v not in pos:
            continue
        ekind = g.edges[u, v].get("kind", "")
        if ekind == "water_branch":
            continue  # 트리는 전기 계통만
        x0, y0 = pos[u]; x1, y1 = pos[v]
        u_kind = g.nodes[u].get("kind", "")
        midy = source_midy.get(u_kind, (y0 + y1) / 2)
        edge_x.extend([x0, x0, x1, x1, None])
        edge_y.extend([y0, midy, midy, y1, None])

    traces = []
    traces.append(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="#9aa5b1", width=1.2),
        hoverinfo="none", showlegend=False,
    ))

    # 1-b) 수도 인입선 trace — 박스 밖 헥사곤에서 박스 하단으로 짧은 점선
    if water_x_list:
        traces.append(go.Scatter(
            x=water_x_list, y=water_y_list, mode="lines",
            line=dict(color="#118ab2", width=1.8, dash="dash"),
            hoverinfo="text", text=["동 수도 인입선"] * len(water_x_list),
            name="수도 인입선", showlegend=True,
        ))

    # 2) 노드 마커
    kinds = {
        "transformer": dict(symbol="diamond", size=24, color="#1d3557"),
        "mdp":         dict(symbol="square",  size=28, color="#2a9d8f"),
        "panel":       dict(symbol="square",  size=18, color="#264653"),
        "store":       dict(symbol="circle",  size=10, color="#3aa55d"),
        "water_main":  dict(symbol="hexagon", size=20, color="#118ab2"),
    }
    for kind, style in kinds.items():
        xs, ys, texts, colors, sizes, lines, widths = [], [], [], [], [], [], []
        for n, data in g.nodes(data=True):
            if data.get("kind") != kind:
                continue
            if n not in pos:
                continue
            x, y = pos[n]
            xs.append(x); ys.append(y)
            label = data.get("label", n)
            extra = ""
            if kind == "store":
                status = (store_status or {}).get(n, "normal")
                colors.append(color_map.get(status, style["color"]))
                sizes.append(style["size"])
                extra = (f"<br>업종: {data.get('category', '?')}"
                         f"<br>임계: {data.get('leak_threshold_mA', '?')} mA")
            elif kind == "panel":
                status = (panel_status or {}).get(n, "normal")
                colors.append(color_map.get(status, style["color"]))
                sizes.append(30 if n == highlight_panel else style["size"])
                extra = f"<br>trip: {data.get('trip_threshold_mA', '?')} mA"
            else:
                colors.append(style["color"])
                sizes.append(style["size"])
            # 테두리: 강조 SDP는 노란 굵은 테두리, 일반 SDP는 동 색상, 나머지는 흰색
            if n == highlight_panel:
                lines.append("#ffb703")
                widths.append(3)
            elif kind == "panel":
                lines.append(_DONG_BORDER.get(data.get("dong", ""), "#264653"))
                widths.append(2.5)
            else:
                lines.append("white")
                widths.append(1)
            texts.append(f"<b>{n}</b><br>{label}{extra}")
        if not xs:
            continue
        traces.append(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(
                symbol=style["symbol"], size=sizes, color=colors,
                line=dict(color=lines, width=widths),
            ),
            text=texts, hoverinfo="text",
            name={"transformer": "변압기", "mdp": "메인분전반(MDP)",
                  "panel": "분전반(SDP)", "store": "점포",
                  "water_main": "동 수도메인"}[kind],
        ))

    # 3) 레벨 라벨 (좌측) + 동 이름 라벨 (SDP 띠 위)
    annotations = []
    label_x = x_left + 0.05  # axis 좌측 안쪽에 우정렬
    store_label_y = -3.0 - (max_rows - 1) * row_dy / 2
    level_rows = [
        (0.0, "한전 변압기"),
        (-0.5, "메인 분전반"),
        (-2.0, "분전반(SDP)"),
        (store_label_y, "점포"),
        (water_y, "동 수도메인"),
    ]
    for ly, lbl in level_rows:
        annotations.append(dict(
            x=label_x, y=ly, xref="x", yref="y",
            text=f"<b>{lbl}</b>", showarrow=False,
            font=dict(size=12, color="#444"), xanchor="left",
            bgcolor="rgba(255,255,255,0.85)",
        ))

    # 동 이름 라벨 — 컬러 띠 안쪽 (중앙)
    for d in market.dongs:
        if not d.panels:
            continue
        panel_xs = [pos[p.panel_id][0] for p in d.panels if p.panel_id in pos]
        if not panel_xs:
            continue
        annotations.append(dict(
            x=(min(panel_xs) + max(panel_xs)) / 2, y=-1.335,
            xref="x", yref="y",
            text=f"<b>{d.name}</b>",
            showarrow=False, xanchor="center", yanchor="middle",
            font=dict(size=11, color="#1a1a1a"),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        xaxis=dict(visible=False, range=[x_left, x_right]),
        yaxis=dict(visible=False, range=[water_y - 0.5, 0.4]),
        margin=dict(l=10, r=10, t=10, b=10),
        height=720, plot_bgcolor="white",
        shapes=shapes,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    bgcolor="rgba(255,255,255,0.85)"),
        annotations=annotations,
    )
    return fig


def _compute_levels(g: nx.DiGraph, market: Market) -> dict[str, int]:
    """0=TR, 1=MDP, 2=panel, 3=store, water_main 별도 (-1)."""
    levels: dict[str, int] = {}
    for n, data in g.nodes(data=True):
        k = data.get("kind")
        if k == "transformer": levels[n] = 0
        elif k == "mdp": levels[n] = 1
        elif k == "panel": levels[n] = 2
        elif k == "store": levels[n] = 3
        elif k == "water_main": levels[n] = -1
    return levels


def _hierarchical_layout(
    g: nx.DiGraph, market: Market, levels: dict[str, int],
) -> tuple[dict, dict]:
    """Top-down tree: TR → MDP → SDP → store(격자) → water_main.

    Returns (pos, info) where info has axis hints (`x_left`, `x_right`,
    `max_store_rows`, `row_dy`, `water_y`).
    """
    pos: dict = {}
    panel_list = list(market.panels)
    n_panels = len(panel_list)

    # 레이아웃 상수 — 한 화면에 18 SDP + 각 SDP 아래 3열 격자가 들어가도록
    sdp_x_gap = 0.18
    store_cols = 3
    store_col_w = 0.05
    row_dy = 0.42

    # SDP 가로 좌표: 등간격, 중앙 정렬
    x_total = (n_panels - 1) * sdp_x_gap if n_panels > 1 else 0.0
    panel_x_arr = [-x_total / 2 + i * sdp_x_gap for i in range(n_panels)]

    # TR / MDP — MDP를 위로 올려 동 박스(top=-1.10)와 간격 확보
    for n, lvl in levels.items():
        if lvl == 0:
            pos[n] = (0.0, 0.0)
        elif lvl == 1:
            pos[n] = (0.0, -0.5)

    # SDPs
    for i, p in enumerate(panel_list):
        pos[p.panel_id] = (panel_x_arr[i], -2.0)

    # 점포 격자: 각 SDP 아래 (3열 × ceil(n/3)행)
    max_rows = 1
    for i, p in enumerate(panel_list):
        n = len(p.stores)
        if n == 0:
            continue
        cols = min(n, store_cols)
        rows = int(np.ceil(n / cols))
        max_rows = max(max_rows, rows)
        px = panel_x_arr[i]
        cluster_w = (cols - 1) * store_col_w
        for k, s in enumerate(p.stores):
            col = k % cols
            row = k // cols
            cx = px - cluster_w / 2 + col * store_col_w
            cy = -3.0 - row * row_dy
            pos[s.store_id] = (cx, cy)

    # 동 수도메인: 점포 영역 아래, 각 dong의 SDP 중앙 x에 정렬
    water_y = -3.0 - (max_rows - 0.5) * row_dy - 0.4
    for d in market.dongs:
        wm = d.main_water_meter_id
        if not d.panels:
            continue
        xs = [panel_x_arr[panel_list.index(p)] for p in d.panels]
        pos[wm] = (sum(xs) / len(xs), water_y)

    # axis 힌트 — 좌측 레벨 라벨 자리 확보 (한글 라벨 ~0.23 단위 + 여유)
    cluster_half = (store_cols - 1) * store_col_w / 2
    x_left = (-x_total / 2) - cluster_half - 0.40
    x_right = (x_total / 2) + cluster_half + 0.22
    info = {
        "x_left": x_left,
        "x_right": x_right,
        "max_store_rows": max_rows,
        "row_dy": row_dy,
        "water_y": water_y,
    }
    return pos, info


# ---------- grouped time-series ----------

def grouped_timeseries_figure(
    series: dict[str, pd.Series],
    thresholds: dict[str, list[tuple[float, str, str]]] | None = None,
    anomaly_spans: list[tuple[pd.Timestamp, pd.Timestamp, str, str]] | None = None,
    title: str = "",
    height: int = 320,
    secondary_y_channels: list[str] | None = None,
) -> go.Figure:
    """Plot multiple series in one panel.

    thresholds[channel] = [(value, color, label), ...]  → horizontal dashed lines
    anomaly_spans = [(start, end, color, hover_text), ...]  → vrect shading
    secondary_y_channels: channels to plot on the right y-axis
    """
    sec = set(secondary_y_channels or [])
    fig = make_subplots(specs=[[{"secondary_y": bool(sec)}]])

    palette = ["#1d3557", "#e63946", "#2a9d8f", "#f4a261", "#264653", "#118ab2"]
    for i, (ch, s) in enumerate(series.items()):
        label = CHANNEL_LABEL.get(ch, ch)
        fig.add_trace(
            go.Scatter(x=s.index, y=s.values, mode="lines", name=label,
                       line=dict(color=palette[i % len(palette)], width=1.6)),
            secondary_y=(ch in sec),
        )

    # threshold lines — 라벨 위치를 분산 (top right / center right / bottom right)
    # x 좌표는 우측 끝 안쪽, y는 dash 라인 살짝 위
    if thresholds:
        all_thresholds = []
        for ch, lines in thresholds.items():
            for v, c, lbl in lines:
                all_thresholds.append((v, c, lbl, ch in sec))
        for value, color, lbl, on_sec in all_thresholds:
            fig.add_hline(
                y=value, line_dash="dash", line_color=color, line_width=1.2,
                secondary_y=on_sec,
            )
            # annotation은 별도 — 우측 끝에 라벨만, 살짝 y 오프셋 줘서 라인과 분리
            fig.add_annotation(
                xref="paper", yref="y2" if on_sec else "y",
                x=1.0, y=value, text=f" {lbl}",
                showarrow=False, xanchor="left", yanchor="middle",
                font=dict(size=10, color=color),
                bgcolor="rgba(255,255,255,0.7)",
            )

    # anomaly shading — Multivariate: 신호가 안 바뀐 차트에서도 명확히 보이도록 진하게.
    # vrect는 면 + 좌/우 경계선까지 그려서 anomaly 구간을 분명히 표시.
    if anomaly_spans:
        for s, e, color, _text in anomaly_spans:
            fig.add_vrect(
                x0=s, x1=e, fillcolor=color, opacity=0.28,
                layer="below",
                line=dict(color="rgba(230,57,70,0.9)", width=1.2, dash="dot"),
            )
        # 각 anomaly span 시작점에 작은 마커를 두어 hover로 정보 보이게
        hov_x = [s for (s, _, _, _) in anomaly_spans]
        hov_text = [t for (_, _, _, t) in anomaly_spans]
        # y는 시리즈의 max에 살짝 못미치는 위치
        if series:
            y_max = max(float(np.nanmax(v.values)) for v in series.values())
            y_min = min(float(np.nanmin(v.values)) for v in series.values())
            y_marker = y_min + (y_max - y_min) * 0.92
            fig.add_trace(go.Scatter(
                x=hov_x, y=[y_marker] * len(hov_x),
                mode="markers",
                marker=dict(symbol="triangle-down", size=10,
                            color="rgba(230,57,70,0.85)",
                            line=dict(width=0.5, color="white")),
                text=hov_text, hoverinfo="text",
                name="이상 시작",
                showlegend=False,
            ))

    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", y=0.97, yanchor="top"),
        height=height,
        margin=dict(l=60, r=140, t=70, b=40),   # 좌/우/하 마진 고정 — 차트 폭 일관
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0.0),
        hovermode="x unified",
    )
    # secondary_y 유무와 무관하게 플롯 영역을 동일하게 — x도메인 강제
    fig.update_xaxes(domain=[0.0, 1.0])
    return fig


# ---------- realistic 2D market plan-view (전주 남부시장) ----------
#
# 좌표계: 동(왼쪽) ←→ 서(오른쪽)는 실제 지도 기준이며, 위가 북(north)이다.
# 실제 남부시장 배치:
#   - 청년몰: 5동 위쪽 별도 구역
#   - 1동: 우측 끝 세로로 긴 블록 (한옥마을 인접)
#   - 2동/3동: 우측 중앙
#   - 5동: 시장 중앙 — 가장 큰 동
#   - 4동/6동: 5동 위/아래
#   - 7~10동: 좌측 컬럼 (북→남: 7→8→9→10)
#   - 전주천: 남쪽(아래) 경계
#   - 풍남문: 동쪽(우측 상단)
#
# 동 블록 좌표 (x0, y0, x1, y1)  — 0.7 단위 gutter로 동 간 간격 확대
# 좌→우 컬럼:  [0.5, 4.5] | g | [5.2, 9.7] | g | [10.4, 14.9] | g | [15.6, 19.6]
# 위→아래 행: [DM 17.1~19.6] | bus/gate (1.5) | [12.6~15.6] | g | [8.9~11.9] | g | [5.7~8.2] | g | [1.0~5.0]
_DONG_RECTS = {
    # 상단 분리 구역 — 청년몰
    "DM":  ( 5.7, 17.1,  10.5, 19.6),

    # 좌측 컬럼 (북→남: 7→8→9→10)
    "D7":  ( 0.5, 12.6,   4.5, 15.6),
    "D8":  ( 0.5,  8.9,   4.5, 11.9),
    "D9":  ( 0.5,  5.7,   4.5,  8.2),
    "D10": ( 0.5,  1.0,   4.5,  5.0),

    # 중앙 컬럼 (북→남: 6→5→4) — 5동이 가장 큼
    "D6":  ( 5.2, 12.6,   9.7, 15.6),
    "D5":  ( 5.2,  5.7,   9.7, 11.9),
    "D4":  ( 5.2,  1.0,   9.7,  5.0),

    # 우측 컬럼 (북→남: 2→3)
    "D2":  (10.4, 12.6,  14.9, 15.6),
    "D3":  (10.4,  5.7,  14.9, 11.9),

    # 1동 — 한옥마을 인접, 우측 세로로 김
    "D1":  (15.6,  5.7,  19.6, 15.6),
}

# 색상 — 실제 남부시장 안내 지도(이미지)의 동별 컬러 코드 참조
_DONG_FILL = {
    "DM":  "rgba(244,143,177,0.55)",   # 청년몰 — 핑크
    "D1":  "rgba(255,167,38,0.55)",    # 1동 — 주황
    "D2":  "rgba(255,138,101,0.55)",   # 2동 — 진주황
    "D3":  "rgba(229,115,115,0.55)",   # 3동 — 빨강
    "D4":  "rgba(149,117,205,0.55)",   # 4동 — 보라
    "D5":  "rgba(63,81,181,0.55)",     # 5동 — 진파랑
    "D6":  "rgba(255,183,77,0.55)",    # 6동 — 주황
    "D7":  "rgba(79,195,247,0.55)",    # 7동 — 파랑
    "D8":  "rgba(240,98,146,0.55)",    # 8동 — 마젠타
    "D9":  "rgba(255,238,88,0.65)",    # 9동 — 노랑
    "D10": "rgba(186,104,200,0.55)",   # 10동 — 자주
}
_PANEL_FILL = "rgba(255,255,255,0.55)"
_STATUS_COLOR = {"normal": "#3aa55d", "alert": "#e63946", "nuisance": "#f4a261"}

def _light_fill(rgba_str: str, alpha: float = 0.13) -> str:
    """기존 `rgba(r,g,b,a)` 문자열에서 알파만 교체."""
    inner = rgba_str.strip()[5:-1]
    parts = [p.strip() for p in inner.split(",")]
    return f"rgba({parts[0]},{parts[1]},{parts[2]},{alpha})"


# 트리 / SDP 마커 테두리용 동 컬러 (불투명, _DONG_FILL의 진한 톤)
_DONG_BORDER = {
    "DM":  "#c2185b",
    "D1":  "#ef6c00",
    "D2":  "#d84315",
    "D3":  "#c62828",
    "D4":  "#5e35b1",
    "D5":  "#1a237e",
    "D6":  "#ef6c00",
    "D7":  "#01579b",
    "D8":  "#ad1457",
    "D9":  "#f9a825",
    "D10": "#6a1b9a",
}


def _market_layout(market: Market):
    """좌표 + 패널 sub-rect 계산. 평면도용.

    - 키가 큰 동(세로>가로)은 위/아래로 분할, 그렇지 않으면 가로로 분할.
    - 수도 메인은 각 동 우상단 코너 내부 (동 이름은 좌상단).
    - TR / MDP는 시장 좌측 상단 외곽.
    """
    pos_store: dict[str, tuple[float, float]] = {}
    pos_panel: dict[str, tuple[float, float]] = {}
    pos_water: dict[str, tuple[float, float]] = {}
    panel_rects: dict[str, tuple[float, float, float, float]] = {}

    for d in market.dongs:
        if d.dong_id not in _DONG_RECTS:
            continue
        x0, y0, x1, y1 = _DONG_RECTS[d.dong_id]
        dong_w = x1 - x0
        dong_h = y1 - y0
        n_panels = len(d.panels)
        split_vertical = dong_h > dong_w  # 세로로 긴 동은 위/아래 분할

        for pi, panel in enumerate(d.panels):
            if split_vertical:
                py0 = y0 + dong_h * pi / n_panels
                py1 = y0 + dong_h * (pi + 1) / n_panels
                inner = (x0 + 0.18, py0 + 0.28, x1 - 0.18, py1 - 0.28)
                pos_panel[panel.panel_id] = (x0 + 0.55, (py0 + py1) / 2)
            else:
                px0 = x0 + dong_w * pi / n_panels
                px1 = x0 + dong_w * (pi + 1) / n_panels
                inner = (px0 + 0.18, y0 + 0.55, px1 - 0.18, y1 - 0.55)
                pos_panel[panel.panel_id] = ((px0 + px1) / 2, y1 - 0.30)
            panel_rects[panel.panel_id] = inner

            # 점포 격자
            n = len(panel.stores)
            w, h = inner[2] - inner[0], inner[3] - inner[1]
            cols = max(1, int(round(np.sqrt(max(1, n * w / max(0.1, h))))))
            rows = int(np.ceil(n / cols))
            cell_w = w / cols
            cell_h = h / rows
            for k, s in enumerate(panel.stores):
                col = k % cols
                row = k // cols
                sx = inner[0] + (col + 0.5) * cell_w
                sy = inner[3] - (row + 0.5) * cell_h
                pos_store[s.store_id] = (sx, sy)

        # 수도 메인: 동 우상단 코너 내부 (동 이름은 좌상단)
        pos_water[d.main_water_meter_id] = (x1 - 0.30, y1 - 0.25)

    # 전기 계통 외부: 시장 좌측, 단선도식 버스바 높이에 MDP
    pos_tr = (-3.0, 18.15)
    pos_mdp = (-3.0, 16.35)
    return pos_store, pos_panel, pos_water, panel_rects, pos_mdp, pos_tr


def market_map_figure(
    market: Market,
    panel_status: dict[str, str] | None = None,
    store_status: dict[str, str] | None = None,
    highlight_panel: str | None = None,
) -> go.Figure:
    """시장 평면도 — 전주 남부시장 실측 배치 + 전기·수도 인입선 + 주변 컨텍스트."""
    pos_store, pos_panel, pos_water, panel_rects, pos_mdp, pos_tr = _market_layout(market)
    fig = go.Figure()

    # 0) 주변 컨텍스트 (도시 배경)
    shapes: list[dict] = []
    annotations: list[dict] = []

    # 전주천 — 남쪽 (아래) 가로 띠
    shapes.append(dict(
        type="rect", x0=-4.5, y0=-1.8, x1=22.5, y1=0.2,
        fillcolor="rgba(129,212,250,0.55)",
        line=dict(color="#4fc3f7", width=1), layer="below",
    ))
    annotations.append(dict(
        x=20.5, y=-0.8, xref="x", yref="y",
        text="<b>전주천 (남쪽)</b>", showarrow=False,
        font=dict(size=13, color="#0277bd"),
        xanchor="right", yanchor="middle",
    ))

    # 한옥마을 — 우측 (동쪽) 영역
    shapes.append(dict(
        type="rect", x0=20.1, y0=0.6, x1=22.3, y1=15.6,
        fillcolor="rgba(215,204,200,0.40)",
        line=dict(color="#a1887f", width=1.2, dash="dot"), layer="below",
    ))
    annotations.append(dict(
        x=21.2, y=8.0, xref="x", yref="y",
        text="<b>한옥마을</b><br>(동쪽)", showarrow=False,
        font=dict(size=13, color="#5d4037"),
        textangle=-90,
    ))

    # 풍남문 — 시장 북쪽 (위)
    annotations.append(dict(
        x=13.0, y=20.7, xref="x", yref="y",
        text="🏯  <b>풍남문</b>", showarrow=True,
        arrowhead=2, arrowwidth=1.5, arrowcolor="#666",
        ax=13.0, ay=19.6, axref="x", ayref="y",
        font=dict(size=14, color="#3e2723"),
        bgcolor="rgba(255,248,225,0.9)", bordercolor="#8d6e63", borderwidth=1,
    ))

    # 1) 동 건물 블록 (배경 색)
    for d in market.dongs:
        if d.dong_id not in _DONG_RECTS:
            continue
        x0, y0, x1, y1 = _DONG_RECTS[d.dong_id]
        shapes.append(dict(
            type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=_DONG_FILL.get(d.dong_id, "rgba(200,200,200,0.4)"),
            line=dict(color="#444", width=2), layer="below",
        ))
        annotations.append(dict(
            x=x0 + 0.25, y=y1 - 0.20, xref="x", yref="y",
            text=f"<b>{d.name}</b> · {d.n_stores}점포",
            showarrow=False, xanchor="left", yanchor="top",
            font=dict(size=15, color="#1a1a1a"),
            bgcolor="rgba(255,255,255,0.78)",
            borderpad=2,
        ))

    # 2) 분전반 sub-rect (점포 군집 경계)
    for panel in market.panels:
        if panel.panel_id not in panel_rects:
            continue
        ix0, iy0, ix1, iy1 = panel_rects[panel.panel_id]
        hl = panel.panel_id == highlight_panel
        shapes.append(dict(
            type="rect", x0=ix0, y0=iy0, x1=ix1, y1=iy1,
            fillcolor=_PANEL_FILL,
            line=dict(color=("#ffb703" if hl else "#999"),
                      width=(3 if hl else 1), dash=("solid" if hl else "dot")),
            layer="below",
        ))

    # 3) 전기 단선도식 — TR → MDP → 가로 버스바 → 컬럼별 vertical riser
    # 3-1. TR → MDP (수직, 굵은 라인)
    fig.add_trace(go.Scatter(
        x=[pos_tr[0], pos_mdp[0]], y=[pos_tr[1], pos_mdp[1]],
        mode="lines", line=dict(color="#1d3557", width=3.5),
        hoverinfo="none", showlegend=False,
    ))
    # 3-2. 가로 버스바 (MDP에서 우측으로) — 시장 위쪽 통로 따라
    y_bus = pos_mdp[1]   # 16.35
    x_bus_end = 20.0
    fig.add_trace(go.Scatter(
        x=[pos_mdp[0], x_bus_end], y=[y_bus, y_bus],
        mode="lines", line=dict(color="#1d3557", width=3),
        hoverinfo="text", text=["MDP 간선버스"] * 2,
        name="MDP 간선버스",
    ))
    # 3-3. 컬럼별 라이저 — 컬럼 중심에서 수직으로 내려/올려옴
    risers = [
        # (riser_x, [dong_ids], direction)
        ( 2.5,  ["D7", "D8", "D9", "D10"], "down"),
        ( 7.45, ["D6", "D5", "D4"],        "down"),
        (12.65, ["D2", "D3"],              "down"),
        (17.6,  ["D1"],                    "down"),
        ( 8.1,  ["DM"],                    "up"),
    ]
    fx, fy = [], []
    for riser_x, dong_ids, direction in risers:
        ys_in_col: list[float] = []
        for did in dong_ids:
            if did in _DONG_RECTS:
                _, y0, _, y1 = _DONG_RECTS[did]
                ys_in_col.extend([y0, y1])
        if not ys_in_col:
            continue
        if direction == "up":  # DM (청년몰)
            fx.extend([riser_x, riser_x, None])
            fy.extend([y_bus, max(ys_in_col), None])
        else:  # 본관 컬럼 — 버스에서 최하단 dong 바닥까지
            fx.extend([riser_x, riser_x, None])
            fy.extend([y_bus, min(ys_in_col), None])
    fig.add_trace(go.Scatter(
        x=fx, y=fy, mode="lines",
        line=dict(color="#5a7a99", width=1.8, dash="dot"),
        hoverinfo="none", name="컬럼 라이저",
    ))

    # 4) 수도 메인 — 전주천에서 각 동으로 (남쪽 트렁크 + 세로 분기)
    y_water_trunk = 0.5
    fig.add_trace(go.Scatter(
        x=[-3.0, 20.0], y=[y_water_trunk, y_water_trunk],
        mode="lines", line=dict(color="#118ab2", width=2.5),
        hoverinfo="text", text=["시청 수도 인입 (전주천 방면)"] * 2,
        name="수도 메인",
    ))
    # 동별 분기: 전기 라이저(컬럼 중심)와 겹치지 않게 동 동쪽 변(east gutter) 사용
    wx, wy = [], []
    branch_x_by_dong = {
        # 좌측 컬럼 → east gutter (x=4.85)
        "D7":  4.85,  "D8":  4.85,  "D9":   4.85,  "D10":  4.85,
        # 중앙 컬럼 → east gutter (x=10.05)
        "D6": 10.05,  "D5": 10.05,  "D4":  10.05,
        # 우측 컬럼 → east gutter (x=15.25)
        "D2": 15.25,  "D3": 15.25,
        # 1동, 청년몰
        "D1": 19.85,  "DM": 10.7,
    }
    for d in market.dongs:
        if d.main_water_meter_id not in pos_water:
            continue
        xm, ym = pos_water[d.main_water_meter_id]
        bx = branch_x_by_dong.get(d.dong_id, xm)
        # 트렁크 → east gutter 위로 → 동 우상단 미터기로 (짧은 좌향 jog)
        wx.extend([bx, bx, xm, None])
        wy.extend([y_water_trunk, ym, ym, None])
    fig.add_trace(go.Scatter(
        x=wx, y=wy, mode="lines",
        line=dict(color="#118ab2", width=1.4, dash="dash"),
        hoverinfo="none", showlegend=False,
    ))

    # 5) 점포 마커 (사각형, 상태별 색)
    s_x, s_y, s_color, s_text = [], [], [], []
    for s in market.stores:
        if s.store_id not in pos_store:
            continue
        x, y = pos_store[s.store_id]
        s_x.append(x); s_y.append(y)
        status = (store_status or {}).get(s.store_id, "normal")
        s_color.append(_STATUS_COLOR[status])
        s_text.append(
            f"<b>{s.store_id}</b><br>업종: {s.profile.category}"
            f"<br>분전반: {s.panel_id}<br>임계: {s.profile.leak_threshold_mA:.0f} mA"
        )
    fig.add_trace(go.Scatter(
        x=s_x, y=s_y, mode="markers",
        marker=dict(symbol="square", size=12, color=s_color,
                    line=dict(color="white", width=1)),
        text=s_text, hoverinfo="text", name="점포",
    ))

    # 6) 분전반 마커 — 선택된 SDP만 라벨, 나머지는 호버만 (겹침 회피)
    p_x, p_y, p_color, p_size, p_line, p_text, p_lbl = [], [], [], [], [], [], []
    for panel in market.panels:
        if panel.panel_id not in pos_panel:
            continue
        x, y = pos_panel[panel.panel_id]
        p_x.append(x); p_y.append(y)
        status = (panel_status or {}).get(panel.panel_id, "normal")
        p_color.append(_STATUS_COLOR[status] if status != "normal" else "#264653")
        hl = panel.panel_id == highlight_panel
        p_size.append(28 if hl else 14)
        p_line.append("#ffb703" if hl else "white")
        p_text.append(
            f"<b>{panel.panel_id}</b><br>{panel.name}"
            f"<br>{len(panel.stores)}점포<br>trip: {panel.trip_threshold_mA:.0f}mA"
        )
        # 라벨은 선택된 SDP만 — 나머지는 hover에서 확인
        p_lbl.append(f"<b>{panel.panel_id}</b>" if hl else "")
    fig.add_trace(go.Scatter(
        x=p_x, y=p_y, mode="markers+text",
        marker=dict(symbol="square", size=p_size, color=p_color,
                    line=dict(color=p_line, width=2)),
        text=p_lbl, textposition="top center",
        textfont=dict(size=12, color="#7a3000"),
        hovertext=p_text, hoverinfo="text", name="분전반(SDP)",
    ))

    # 7) MDP / 변압기 마커
    fig.add_trace(go.Scatter(
        x=[pos_mdp[0]], y=[pos_mdp[1]], mode="markers+text",
        marker=dict(symbol="square", size=32, color="#2a9d8f",
                    line=dict(color="white", width=2)),
        text=["MDP"], textposition="bottom center",
        hovertext=f"메인 분전반 {market.mdp_id}<br>trip: {market.mdp_trip_threshold_mA:.0f} mA",
        hoverinfo="text", name="메인분전반(MDP)",
    ))
    fig.add_trace(go.Scatter(
        x=[pos_tr[0]], y=[pos_tr[1]], mode="markers+text",
        marker=dict(symbol="diamond", size=24, color="#1d3557",
                    line=dict(color="white", width=2)),
        text=["TR"], textposition="bottom center",
        hovertext=f"한전 변압기 {market.transformer_id}",
        hoverinfo="text", name="변압기",
    ))

    # 8) 수도 메인 마커
    w_x, w_y, w_text = [], [], []
    for d in market.dongs:
        if d.main_water_meter_id not in pos_water:
            continue
        x, y = pos_water[d.main_water_meter_id]
        w_x.append(x); w_y.append(y)
        w_text.append(f"<b>{d.main_water_meter_id}</b><br>{d.name} 수도 메인")
    fig.add_trace(go.Scatter(
        x=w_x, y=w_y, mode="markers",
        marker=dict(symbol="hexagon", size=18, color="#118ab2",
                    line=dict(color="white", width=2)),
        text=w_text, hoverinfo="text", name="수도 메인",
    ))

    # 9) 게이트 (실제 시장 안내도 참고 — 주요 출입구만)
    gate_annots = [
        # 남문-1 — 청년몰과 본관 사이 east gutter, 버스라인 위쪽
        dict(x=4.85, y=16.0, text="🚪 남문-1"),
        # 북문-1, 북문-2 — 시장 남쪽 (전주천 방면)
        dict(x=2.5, y=0.6, text="🚪 북문-2"),
        dict(x=12.65, y=0.6, text="🚪 북문-1"),
        # 서문 — 한옥마을 방면 (1동 우측)
        dict(x=19.85, y=10.0, text="🚪 서문"),
    ]
    for g in gate_annots:
        annotations.append(dict(
            x=g["x"], y=g["y"], xref="x", yref="y",
            text=f"<b>{g['text']}</b>", showarrow=False,
            font=dict(size=11, color="#5d4037"),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#8d6e63", borderwidth=0.6,
            xanchor="center", yanchor="middle",
        ))

    # 10) 나침반 (좌하단 외곽)
    annotations.append(dict(
        x=-3.7, y=2.5, xref="x", yref="y",
        text="<b>N</b><br>↑<br>S", showarrow=False,
        font=dict(size=15, color="#333"),
        bgcolor="rgba(255,255,255,0.92)", bordercolor="#666", borderwidth=1,
    ))

    fig.update_layout(
        xaxis=dict(visible=False, range=[-4.7, 23.0]),
        yaxis=dict(visible=False, range=[-2.0, 21.6],
                   scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=10, b=10),
        height=860, plot_bgcolor="#f4f1ea",
        shapes=shapes, annotations=annotations,
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    xanchor="left", x=0.0,
                    bgcolor="rgba(255,255,255,0.85)",
                    font=dict(size=11)),
    )
    return fig
