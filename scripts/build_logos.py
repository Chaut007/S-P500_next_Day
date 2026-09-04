"""Phase 2 -- fetch the company marks used by the leaderboard on the home page.

Simple Icons ships one monochrome glyph per brand as a single SVG path with no
fill of its own, which is exactly what a dark dashboard wants: recolouring is a
one-attribute edit and every mark ends up the same weight and size. Marks are
downloaded once and committed, so the dashboard itself never reaches the
network.

Seven of the twenty-three companies that have held a top-ten slot have no glyph
in the set -- the industrials, the insurers and the two consumer names all
trade under wordmarks rather than symbols. They are left without a mark on
purpose. Drawing their ticker into the empty slot was tried and read as a
duplicate, because the chart already prints the ticker beside every row: the
gutter said "XOM  XOM", and the drawn wordmark was three times the width of a
glyph, so the rows no longer lined up. A blank costs nothing, since the ticker
is what identifies the row either way.

The marks are trademarks of their owners and are used here to identify the
companies in an academic study, which is what nominative use covers.

Run from the project root:
    python -m scripts.build_logos
    python -m scripts.build_logos --colour "#EDF0F7"
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request

from src.config import LOGOS_DIR
from src.logger import get_logger

log = get_logger("build_logos")

CDN = "https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/{}.svg"

# Ticker -> Simple Icons slug. GOOG and GOOGL share Google's glyph; they are
# separate rank slots and the chart labels them by ticker to tell them apart.
SLUGS = {
    "AAPL": "apple",
    "AMZN": "amazon",
    "AVGO": "broadcom",
    "BA": "boeing",
    "GOOG": "google",
    "GOOGL": "google",
    "INTC": "intel",
    "JPM": "chase",
    "META": "meta",
    "MSFT": "microsoft",
    "NVDA": "nvidia",
    "T": "atandt",
    "TSLA": "tesla",
    "V": "visa",
    "VZ": "verizon",
    "WMT": "walmart",
}

# In the top ten at some point but absent from Simple Icons. Listed so the gap
# is a recorded decision rather than something that looks like an oversight.
NO_GLYPH = ["BRK-B", "CVX", "JNJ", "LLY", "PG", "UNH", "XOM"]

DEFAULT_COLOUR = "#EDF0F7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and tint company marks")
    parser.add_argument("--colour", default=DEFAULT_COLOUR,
                        help="fill applied to every mark")
    parser.add_argument("--force", action="store_true",
                        help="re-download marks that are already present")
    return parser.parse_args()


def tint(svg: str, colour: str) -> str:
    """Give the root <svg> a fill so the path inherits it.

    Simple Icons paths carry no fill and therefore render black by default,
    which is invisible on this background.
    """
    if "<svg" not in svg:
        raise ValueError("not an SVG document")
    # Drop any fill already on the root, then set ours.
    head_end = svg.index(">", svg.index("<svg"))
    head = re.sub(r'\s+fill="[^"]*"', "", svg[:head_end])
    return f'{head} fill="{colour}"{svg[head_end:]}'


def fetch(slug: str) -> str:
    request = urllib.request.Request(CDN.format(slug),
                                     headers={"User-Agent": "sp555-build-logos"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8")


def main() -> int:
    args = parse_args()
    LOGOS_DIR.mkdir(parents=True, exist_ok=True)

    written = skipped = 0

    for ticker, slug in SLUGS.items():
        path = LOGOS_DIR / f"{ticker}.svg"
        if path.exists() and not args.force:
            skipped += 1
            continue
        try:
            svg = tint(fetch(slug), args.colour)
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
            log.error("Could not fetch %s (%s): %s", ticker, slug, exc)
            return 1
        path.write_text(svg, encoding="utf-8")
        log.info("%-6s <- simple-icons/%s", ticker, slug)
        written += 1

    # Clear any mark left behind for a company that is meant to have none, so a
    # rerun after this list changes does not leave a stale file being served.
    for ticker in NO_GLYPH:
        stale = LOGOS_DIR / f"{ticker}.svg"
        if stale.exists():
            stale.unlink()
            log.info("%-6s removed (no glyph in Simple Icons)", ticker)

    log.info("Marks in %s: %d written, %d already present, %d without a glyph",
             LOGOS_DIR, written, skipped, len(NO_GLYPH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
