"""Market / Dong / Panel / Store topology (Plan §2 + 분전반 레이어).

계층:
  TR-MAIN (한전 변압기)
   └─ MDP-01 (시장 메인 분전반)
       └─ SDP_<dong>_<idx> (동별 분전반)
           └─ store
  수도는 별도: WM_<dong> (동 수도 메인) → store branch
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import networkx as nx
import numpy as np
import yaml


@dataclass
class StoreProfile:
    category: str
    leak_threshold_mA: float
    load_kw_day: float
    load_kw_night: float
    water_lpm_day: float
    water_lpm_night: float
    has_inverter: bool
    daily_open_hour: int
    daily_close_hour: int
    seasonal_boost_months: list[int] = field(default_factory=list)


@dataclass
class Store:
    store_id: str
    dong_id: str
    panel_id: str
    profile: StoreProfile
    branch_length_m: float


@dataclass
class Panel:
    """SDP (Sub-Distribution Panel) — 동 내부의 분전반."""
    panel_id: str
    name: str
    dong_id: str
    trip_threshold_mA: float
    stores: list[Store] = field(default_factory=list)


@dataclass
class Dong:
    dong_id: str
    name: str
    main_water_meter_id: str
    main_length_m: float
    panel_trip_threshold_mA: float
    panels: list[Panel] = field(default_factory=list)

    @property
    def stores(self) -> Iterator[Store]:
        for p in self.panels:
            yield from p.stores

    @property
    def n_stores(self) -> int:
        return sum(len(p.stores) for p in self.panels)


@dataclass
class Market:
    name: str
    season: str
    start_date: str
    transformer_id: str
    mdp_id: str
    mdp_trip_threshold_mA: float
    mnf_threshold_lpm_per_km: float
    dongs: list[Dong]
    graph: nx.DiGraph
    profiles: dict[str, StoreProfile]
    noise_cfg: dict
    env_cfg: dict
    weekly_cfg: dict
    leak_baseline_cfg: dict

    @property
    def stores(self) -> Iterator[Store]:
        for d in self.dongs:
            yield from d.stores

    @property
    def panels(self) -> Iterator[Panel]:
        for d in self.dongs:
            yield from d.panels

    def store_by_id(self, sid: str) -> Store:
        for s in self.stores:
            if s.store_id == sid:
                return s
        raise KeyError(sid)

    def panel_by_id(self, pid: str) -> Panel:
        for p in self.panels:
            if p.panel_id == pid:
                return p
        raise KeyError(pid)


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _draw_categories(n: int, mix: list[dict], rng: np.random.Generator) -> list[dict]:
    weights = np.array([m["weight"] for m in mix], dtype=float)
    weights /= weights.sum()
    idx = rng.choice(len(mix), size=n, p=weights)
    return [mix[i] for i in idx]


def _split_count(total: int, n_groups: int) -> list[int]:
    """Distribute `total` across `n_groups` as evenly as possible."""
    base = total // n_groups
    rem = total - base * n_groups
    return [base + (1 if i < rem else 0) for i in range(n_groups)]


def build_market(config_dir: str | Path, seed: int = 42) -> Market:
    config_dir = Path(config_dir)
    topo = _load_yaml(config_dir / "market_topology.yaml")
    prof_cfg = _load_yaml(config_dir / "store_profiles.yaml")
    rng = np.random.default_rng(seed)

    mix = prof_cfg["mix"]
    profile_objs: dict[str, StoreProfile] = {}
    for m in mix:
        profile_objs[m["category"]] = StoreProfile(
            category=m["category"],
            leak_threshold_mA=float(m["leak_threshold_mA"]),
            load_kw_day=float(m["load_kw_day"]),
            load_kw_night=float(m["load_kw_night"]),
            water_lpm_day=float(m["water_lpm_day"]),
            water_lpm_night=float(m["water_lpm_night"]),
            has_inverter=bool(m["has_inverter"]),
            daily_open_hour=int(m["daily_open_hour"]),
            daily_close_hour=int(m["daily_close_hour"]),
            seasonal_boost_months=list(m.get("seasonal_boost_months", [])),
        )

    g = nx.DiGraph()
    tr_id = topo["market"]["transformer_id"]
    mdp_id = topo["market"]["mdp_id"]
    g.add_node(tr_id, kind="transformer", label="한전 변압기")
    g.add_node(mdp_id, kind="mdp", label="메인 분전반")
    g.add_edge(tr_id, mdp_id, kind="hv_to_mdp")

    dongs: list[Dong] = []
    for d in topo["dongs"]:
        dong = Dong(
            dong_id=d["id"],
            name=d["name"],
            main_water_meter_id=d["main_water_meter"],
            main_length_m=float(topo["market"]["pipe_main_length_m"]),
            panel_trip_threshold_mA=float(d.get("panel_trip_threshold_mA", 100.0)),
        )
        n_panels = int(d.get("n_panels", 1))
        split = _split_count(int(d["n_stores"]), n_panels)
        drawn_all = _draw_categories(int(d["n_stores"]), mix, rng)
        cursor = 0
        for pi, count in enumerate(split, start=1):
            pid = f"SDP_{d['id']}_{pi}"
            panel = Panel(
                panel_id=pid,
                name=f"{d['name']} {pi}번 분전반",
                dong_id=d["id"],
                trip_threshold_mA=dong.panel_trip_threshold_mA,
            )
            g.add_node(pid, kind="panel", dong=d["id"], label=panel.name,
                       trip_threshold_mA=panel.trip_threshold_mA)
            g.add_edge(mdp_id, pid, kind="mdp_to_sdp")
            for k in range(count):
                cat = drawn_all[cursor]; cursor += 1
                sid = f"{d['id']}-{(cursor):03d}"
                store = Store(
                    store_id=sid, dong_id=d["id"], panel_id=pid,
                    profile=profile_objs[cat["category"]],
                    branch_length_m=float(topo["market"]["pipe_branch_length_m"]),
                )
                panel.stores.append(store)
                g.add_node(sid, kind="store", dong=d["id"], panel=pid,
                           category=cat["category"],
                           leak_threshold_mA=profile_objs[cat["category"]].leak_threshold_mA)
                g.add_edge(pid, sid, kind="sdp_to_store")
            dong.panels.append(panel)

        # 수도 메인 노드 (전기와는 별도 시스템)
        wm = dong.main_water_meter_id
        g.add_node(wm, kind="water_main", dong=d["id"],
                   label=f"{d['name']} 수도 메인")
        for s in dong.stores:
            g.add_edge(wm, s.store_id, kind="water_branch")

        dongs.append(dong)

    return Market(
        name=topo["market"]["name"],
        season=topo["market"]["season"],
        start_date=topo["market"]["start_date"],
        transformer_id=tr_id,
        mdp_id=mdp_id,
        mdp_trip_threshold_mA=float(topo["market"]["mdp_trip_threshold_mA"]),
        mnf_threshold_lpm_per_km=float(topo["market"].get("mnf_threshold_lpm_per_km", 1.0)),
        dongs=dongs, graph=g, profiles=profile_objs,
        noise_cfg=prof_cfg["noise"], env_cfg=prof_cfg["environment"],
        weekly_cfg=prof_cfg["weekly"], leak_baseline_cfg=prof_cfg["leakage_baseline_mA"],
    )


def summary(market: Market) -> dict:
    cat_count: dict[str, int] = {}
    for s in market.stores:
        cat_count[s.profile.category] = cat_count.get(s.profile.category, 0) + 1
    return {
        "name": market.name,
        "season": market.season,
        "dongs": [(d.dong_id, d.n_stores, len(d.panels)) for d in market.dongs],
        "total_stores": sum(d.n_stores for d in market.dongs),
        "total_panels": sum(len(d.panels) for d in market.dongs),
        "categories": cat_count,
    }
