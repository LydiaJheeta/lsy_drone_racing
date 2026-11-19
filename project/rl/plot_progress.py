"""Plot the PPO training curves from a `progress.csv` written by `rl.ppo`.

Usage:
    cd project
    env -u PYTHONPATH pixi run -e tests --frozen python -m rl.plot_progress \
        --run=results/ppo_<ts>
"""

from __future__ import annotations

import csv
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PANELS = [
    ("ep_reward_mean", "mean episode reward", None),
    ("gates_mean", "mean gates passed", (0, 5)),
    ("frac_success", "episodes ending in success", (0, 1)),
    ("frac_terminated_fail", "episodes ending in a crash", (0, 1)),
    ("sps_loop", "steps/s (env + policy + update)", None),
    ("entropy", "policy entropy", None),
]


def _read(path: Path) -> dict[str, np.ndarray]:
    """Read progress.csv into float columns (empty cells become NaN)."""
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    out: dict[str, np.ndarray] = {}
    for key in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[key]))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        out[key] = np.asarray(vals)
    return out


def plot(run: str, out: str | None = None) -> Path:
    """Write `curves.png` next to the run's `progress.csv`."""
    run_dir = Path(run)
    data = _read(run_dir / "progress.csv")
    x = data["global_step"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    for ax, (key, label, ylim) in zip(axes.ravel(), PANELS, strict=True):
        y = data[key]
        ax.plot(x, y, lw=1.0, color="tab:blue")
        if np.isfinite(y).sum() > 10:  # running mean over ~10 updates
            w = max(3, len(y) // 25)
            kernel = np.ones(w) / w
            pad = np.convolve(np.nan_to_num(y, nan=np.nanmean(y)), kernel, mode="same")
            ax.plot(x, pad, lw=1.8, color="tab:orange", label=f"mean of {w}")
            ax.legend(loc="best", fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
    for ax in axes[-1]:
        ax.set_xlabel("environment steps")
    fig.suptitle(f"PPO gate-progress, {run_dir.name}", fontsize=11)
    fig.tight_layout()
    path = Path(out) if out else run_dir / "curves.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"Wrote {path}")
    return path


if __name__ == "__main__":
    fire.Fire({"plot": plot})
