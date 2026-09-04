"""The animated leaderboard on the home page.

This replaces a pre-rendered MP4. A video has to be watched from the start to
reach any particular month, cannot be inspected, and had to be rebuilt by a
separate script whenever the data changed. The figure below carries one frame
per month end and a slider, so a reader can go straight to March 2020 or drag
across the decade at their own pace, and it inherits the dashboard's theme
instead of arriving as a white rectangle in the middle of a dark page.

Bars are coloured by sector rather than by company, which is what lets the
animation make its argument without labels: energy and industrials drain out of
the block while technology fills it.

Company marks are served from app/static/logos and referenced by URL. A base64
copy in each of 120 frames would run to megabytes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lib import MUTED, NOTE, SECTOR_COLORS, SECTOR_FALLBACK, TEXT
from src.config import LOGOS_DIR

# Streamlit serves app/static at this prefix. Absolute so the URL resolves the
# same way regardless of which page path the browser is sitting on.
LOGO_URL = "/app/static/logos/{}.svg"

# Geometry as a fraction of the largest market cap in the window, so the layout
# holds at any figure size. Negative x is a gutter the bars never enter; it
# holds the marks. The tickers are y axis tick labels rather than annotations
# placed by hand: an annotation has to be anchored at a fixed x, and a
# right-anchored one grows leftward, so "GOOGL" ran back over its own glyph.
# Tick labels get their own margin from automargin at whatever width they need.
_MARK_X = -0.07
_LEFT = -0.14
_RIGHT = 1.10
_MARK_W = 0.10
_MARK_H = 0.80

_FRAME_MS = 260


def _round_ticks(peak: float, target: int = 5) -> list[float]:
    """Gridline values at round numbers rather than even fractions of the peak.

    Dividing the peak into four gives ticks like 1,222 and 3,667, which read as
    data even though they are only the axis.
    """
    rough = peak / target
    magnitude = 10.0 ** np.floor(np.log10(rough))
    step = next(m for m in (1, 2, 2.5, 5, 10) if magnitude * m >= rough) * magnitude
    return list(np.arange(0, peak, step))


def _logo(ticker: str) -> str | None:
    return LOGO_URL.format(ticker) if (LOGOS_DIR / f"{ticker}.svg").exists() else None


def _frame_parts(row: pd.Series, meta: pd.DataFrame, top_n: int, peak: float,
                 stamp: pd.Timestamp):
    """Build the bar, marks and labels for a single month."""
    # Smallest first: Plotly's y axis counts upward, so position 0 is the bottom
    # of the chart and the leader ends up on top.
    top = row.dropna().nlargest(top_n).iloc[::-1]
    tickers = list(top.index)
    values = top.to_numpy(dtype="float64")
    positions = np.arange(len(tickers))

    sectors = [meta["sector"].get(t, "") for t in tickers]
    names = [meta["name"].get(t, t) for t in tickers]

    bar = go.Bar(
        x=values,
        y=positions,
        orientation="h",
        marker=dict(
            color=[SECTOR_COLORS.get(s, SECTOR_FALLBACK) for s in sectors],
            line=dict(width=0),
        ),
        text=[f"{v:,.0f}" for v in values],
        textposition="outside",
        textfont=dict(color=MUTED, size=11),
        cliponaxis=False,
        customdata=np.column_stack([names, sectors]),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                      "<br>%{x:,.0f} bn USD<extra></extra>",
        showlegend=False,
    )

    images = []
    for position, ticker in zip(positions, tickers):
        source = _logo(ticker)
        if source is None:
            continue
        images.append(go.layout.Image(
            source=source,
            xref="x", yref="y",
            x=_MARK_X * peak, y=float(position),
            sizex=_MARK_W * peak, sizey=_MARK_H,
            xanchor="center", yanchor="middle",
            sizing="contain", layer="above",
        ))

    # The month is the one thing a reader tracks while the animation runs, so it
    # is set large and low-contrast rather than tucked into an axis title.
    annotations = [go.layout.Annotation(
        x=0.99, y=0.04, xref="paper", yref="paper",
        text=stamp.strftime("%b %Y"), showarrow=False,
        xanchor="right", yanchor="bottom",
        font=dict(color=NOTE, size=30, family="system-ui, sans-serif"),
    )]

    # Seven of the twenty-three companies have no glyph, so for those rows the
    # tick label is the only identifier. It also separates GOOG from GOOGL,
    # which share Alphabet's mark.
    ticks = dict(tickvals=[float(p) for p in positions], ticktext=tickers)

    return bar, images, annotations, ticks


def _sector_legend(sectors: list[str]) -> list[go.Scatter]:
    """Legend-only traces, one per sector present in the window.

    The bars are a single trace with per-point colours, so they cannot carry a
    legend of their own. These sit after it and the frames never touch them.
    """
    return [
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=9, color=SECTOR_COLORS.get(s, SECTOR_FALLBACK),
                        symbol="square"),
            name=s, showlegend=True, hoverinfo="skip",
        )
        for s in sectors
    ]


def leaderboard_figure(caps: pd.DataFrame, meta: pd.DataFrame,
                       top_n: int = 10, freq: str = "ME") -> go.Figure:
    """An animated top-N bar chart with a month slider.

    `caps` is date-indexed market cap in USD billions, one column per ticker.
    `meta` is indexed by ticker and supplies name and sector.
    """
    monthly = caps.resample(freq).last().dropna(how="all")
    peak = float(np.nanmax(monthly.to_numpy()))

    frames, steps, first = [], [], None
    for stamp, row in monthly.iterrows():
        bar, images, annotations, ticks = _frame_parts(row, meta, top_n, peak, stamp)
        if first is None:
            first = (bar, images, annotations, ticks)
        name = stamp.strftime("%Y-%m")
        frames.append(go.Frame(
            name=name,
            data=[bar],
            traces=[0],
            layout=go.Layout(images=images, annotations=annotations,
                             yaxis=dict(**ticks)),
        ))
        # Every step is labelled. Plotly thins the labels to whatever fits, and
        # it chooses by position, not by which ones carry text -- labelling only
        # the Decembers left the rail blank, because the steps it kept were the
        # ones with an empty string.
        steps.append(dict(
            label=stamp.strftime("%b %Y"),
            method="animate",
            args=[[name], dict(mode="immediate",
                               frame=dict(duration=0, redraw=True),
                               transition=dict(duration=0))],
        ))

    bar, images, annotations, ticks = first

    # Only the sectors that actually reach the top ten, in the order they carry
    # weight, so the legend is not padded with sectors that never appear.
    seen = [s for s in SECTOR_COLORS if s in set(meta["sector"].dropna())]

    figure = go.Figure(
        data=[bar, *_sector_legend(seen)],
        frames=frames,
    )

    figure.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=8, b=8),
        images=images,
        annotations=annotations,
        bargap=0.28,
        xaxis=dict(
            range=[_LEFT * peak, _RIGHT * peak],
            showgrid=True, gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            # The gutter left of zero holds the marks, so ticks are pinned to
            # the positive side instead of running under the labels.
            tickvals=_round_ticks(peak),
            ticktext=[f"{v:,.0f}" for v in _round_ticks(peak)],
            title=dict(text="Market capitalisation (USD billions)",
                       font=dict(color=NOTE, size=11)),
        ),
        yaxis=dict(
            range=[-0.7, top_n - 0.3],
            showgrid=False, zeroline=False,
            automargin=True, ticks="",
            tickfont=dict(color=MUTED, size=12),
            **ticks,
        ),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0,
                    font=dict(color=MUTED, size=11)),
        sliders=[dict(
            active=0,
            x=0, y=0, xanchor="left", yanchor="top",
            pad=dict(t=48, b=6, l=2),
            len=1.0,
            currentvalue=dict(visible=False),
            transition=dict(duration=0),
            bgcolor="rgba(255,255,255,0.14)",
            bordercolor="rgba(255,255,255,0.14)",
            activebgcolor=TEXT,
            tickcolor="rgba(255,255,255,0.22)",
            font=dict(color=MUTED, size=10),
            steps=steps,
        )],
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0, y=0, xanchor="left", yanchor="top",
            pad=dict(t=6, l=2),
            showactive=False,
            bgcolor="rgba(255,255,255,0.07)",
            bordercolor="rgba(255,255,255,0.16)",
            font=dict(color=TEXT, size=12),
            buttons=[
                dict(label="▶  Play", method="animate",
                     args=[None, dict(mode="immediate", fromcurrent=True,
                                      frame=dict(duration=_FRAME_MS, redraw=True),
                                      transition=dict(duration=_FRAME_MS // 2,
                                                      easing="cubic-in-out"))]),
                dict(label="❚❚  Pause", method="animate",
                     args=[[None], dict(mode="immediate",
                                        frame=dict(duration=0, redraw=False),
                                        transition=dict(duration=0))]),
            ],
        )],
    )
    return figure
