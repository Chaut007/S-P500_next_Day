"""Phase 2 -- render the bar chart race used on the dashboard home page.

The animation is the study in one picture: Exxon, GE and Johnson & Johnson slide
out of the top ten while Nvidia, Tesla and Meta climb in, and the leading bars
stretch further ahead every year as the index concentrates.

Rendering happens once, offline. Streamlit then serves a static file, which keeps
the home page instant instead of recomputing 500 tickers on every page load.

Run from the project root:
    python -m scripts.build_race
    python -m scripts.build_race --steps 8 --fps 30
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ASSETS_DIR, PROCESSED_DIR, ensure_dirs, load_config
from src.logger import get_logger
from src.utils import load_table

log = get_logger("build_race")

OUTPUT_PATH = ASSETS_DIR / "top10_race.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the top-N bar chart race")
    parser.add_argument("--steps", type=int, default=6,
                        help="interpolated frames between month ends")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    return parser.parse_args()


def load_market_caps() -> pd.DataFrame:
    """Date-indexed market caps for the whole universe, in USD billions."""
    caps = load_table(PROCESSED_DIR / "mcap_daily.parquet")
    caps["date"] = pd.to_datetime(caps["date"])
    return caps.set_index("date").sort_index()


def build_frames(caps: pd.DataFrame, freq: str, steps: int) -> pd.DataFrame:
    """Month-end snapshots, linearly interpolated into `steps` frames each.

    Daily data would be roughly 2,500 frames, far more than an animation needs.
    Interpolating between month ends keeps the bars sliding smoothly while a
    company overtakes another instead of jumping between positions.
    """
    monthly = caps.resample(freq).last().dropna(how="all")
    log.info("Month-end snapshots: %d", len(monthly))

    if steps <= 1:
        return monthly

    # Build a dense index, then interpolate the whole matrix at once so that
    # ranking is recomputed on smoothly moving values.
    dense_index = pd.date_range(
        monthly.index.min(), monthly.index.max(),
        periods=(len(monthly) - 1) * steps + 1,
    )
    dense = (
        monthly.reindex(monthly.index.union(dense_index))
        .interpolate(method="index", limit_area="inside")
        .reindex(dense_index)
    )
    log.info("Interpolated frames: %d", len(dense))
    return dense


def colour_map(tickers: list[str]) -> dict[str, tuple]:
    """Stable colour per ticker so a bar keeps its colour as it moves."""
    palette = plt.get_cmap("tab20")
    return {t: palette(i % 20) for i, t in enumerate(sorted(tickers))}


def render_race(
    frames: pd.DataFrame,
    top_n: int,
    output_path,
    fps: int,
    names: dict[str, str] | None = None,
) -> None:
    """Write the animation to MP4."""
    import imageio.v2 as imageio

    colours = colour_map(list(frames.columns))
    names = names or {}

    # A fixed axis limit would waste most of the frame early on, so the limit
    # follows the leader with a little headroom.
    peak = float(np.nanmax(frames.to_numpy()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, macro_block_size=None)

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=110)

    try:
        for i, (timestamp, row) in enumerate(frames.iterrows()):
            top = row.dropna().nlargest(top_n).iloc[::-1]  # smallest first for barh

            ax.clear()
            positions = np.arange(len(top))
            ax.barh(
                positions,
                top.to_numpy(),
                color=[colours.get(t, "#888888") for t in top.index],
                edgecolor="none",
            )

            ax.set_yticks(positions)
            ax.set_yticklabels([names.get(t, t) for t in top.index], fontsize=11)
            ax.set_xlim(0, peak * 1.08)
            ax.set_xlabel("Market capitalisation (USD billions)", fontsize=10)
            ax.set_title("S&P 500 — largest constituents by market cap",
                         fontsize=14, pad=14, loc="left")

            for pos, value, ticker in zip(positions, top.to_numpy(), top.index):
                ax.text(value + peak * 0.008, pos, f"{value:,.0f}",
                        va="center", fontsize=9, color="#333333")

            ax.text(0.98, 0.06, timestamp.strftime("%b %Y"),
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=26, color="#bbbbbb", weight="bold")

            ax.grid(axis="x", alpha=0.25)
            ax.set_axisbelow(True)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            fig.tight_layout()

            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[..., :3]

            # libx264 with yuv420p needs both dimensions even; matplotlib is
            # happy to hand back an odd number of pixels and ffmpeg then dies
            # with a broken pipe rather than a useful message.
            height, width = image.shape[:2]
            image = image[: height - height % 2, : width - width % 2]

            writer.append_data(image)

            if (i + 1) % 100 == 0:
                log.info("Rendered %d/%d frames", i + 1, len(frames))
    finally:
        writer.close()
        plt.close(fig)

    log.info("Animation written: %s", output_path)


def main() -> int:
    args = parse_args()
    cfg = load_config()
    ensure_dirs()

    dash_cfg = cfg["dashboard"]
    top_n = args.top_n or dash_cfg["race_top_n"]
    fps = args.fps or dash_cfg["race_fps"]

    caps = load_market_caps()
    frames = build_frames(caps, dash_cfg["race_freq"], args.steps)

    names: dict[str, str] = {}
    try:
        universe = load_table(PROCESSED_DIR / "universe.parquet")
        names = dict(zip(universe["ticker"], universe["ticker"]))
    except FileNotFoundError:
        log.warning("universe table not found; bars will be labelled by ticker")

    render_race(frames, top_n, OUTPUT_PATH, fps, names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
