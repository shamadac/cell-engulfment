from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from matplotlib.axes import Axes

from diagnostics import plot_size_histogram


def test_plot_size_histogram_creates_file_and_threshold_lines(tmp_path: Path, monkeypatch) -> None:
    calls: list[float] = []
    original_axvline = Axes.axvline

    def spy(self, x, *args, **kwargs):
        calls.append(float(x))
        return original_axvline(self, x, *args, **kwargs)

    monkeypatch.setattr(Axes, "axvline", spy)

    config = SimpleNamespace(dpi=300, format="png")
    df = pd.DataFrame({"Volume (micron^3)": [0.1, 0.2, 0.3, 0.4, 0.5]})
    output_path = plot_size_histogram(df, "hflu", "A1", 0.2, 0.5, tmp_path, config)

    assert output_path.exists()
    assert 0.2 in calls
    assert 0.5 in calls
