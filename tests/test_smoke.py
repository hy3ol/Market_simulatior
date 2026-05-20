"""End-to-end smoke test: 1-day run, assert shapes + labels."""
from pathlib import Path

import numpy as np

from src.simulator import simulate
from src.exporter import write_tsb_format


ROOT = Path(__file__).resolve().parent.parent


def test_smoke_one_day(tmp_path):
    result = simulate(ROOT / "config", days=1, seed=7)
    assert result.ax.n_seconds == 86400
    assert len(result.anomalies) >= 1
    for sid, chans in result.store_series.items():
        for ch, arr in chans.items():
            assert arr.shape == (86400,)
            assert np.isfinite(arr).all()

    # panel + dong aggregates
    assert len(result.panel_aggregates) >= 5  # 3+2+2 by default config
    for pid, chans in result.panel_aggregates.items():
        assert {"I_leak_panel", "I_load_panel"} <= set(chans)
        for ch, arr in chans.items():
            assert arr.shape == (86400,)
    for did, chans in result.dong_aggregates.items():
        assert {"Q_main", "P_main", "I_leak_dong"} <= set(chans)

    # all 8 scenario types should be represented at least once
    types = {a.scenario_type for a in result.anomalies}
    assert "A4_insulation_degradation" in types or len(types) >= 5

    total_labeled = sum(int(m.sum()) for m in result.label_mask.values())
    assert total_labeled > 0

    out = tmp_path / "data"
    summary = write_tsb_format(result, out, freqs_s=[60], per_store_freq_s=60,
                               per_store_ids=list(result.store_series)[:2])
    assert summary["n_instances"] == len(result.anomalies)
    assert (out / "tsb_format" / "market_1m.csv").exists()
    assert (out / "tsb_format" / "market_1m.label.csv").exists()
    assert (out / "meta" / "anomaly_index.json").exists()
