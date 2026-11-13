#!/usr/bin/env python3
"""Forecast the next hours of Polymarket's "Next CEO of NVIDIA?" Yes prices.

Reads every <id>.in case from $MAKEFUN_INPUT_DIR (default /input) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (default /output). For each market it forecasts
a random-walk spread around the checkpoint price, with the per-minute spread
estimated from the last 120 one-minute changes:

    sigma = min(0.005, max(0.0002, 0.55 * rms(d), 0.0025 * sqrt(p * (1 - p))))
    q_tau(h) = clamp(x + z_tau * sqrt(h) * sigma, 0, 1)

rounded to 6 decimals; the floor rises with the price level. Python 3 standard library only, no network, no randomness.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
MAX_CHANGES = 120
SIGMA_FLOOR = 0.0002
SIGMA_CAP = 0.005
RMS_SCALE = 0.55
PRICE_SCALE = 0.0025
HORIZON_MINUTES = 240


def clamp6(value: float) -> float:
    """Clamp to [0, 1], round to 6 decimals, and turn -0.0 into 0.0."""
    if not math.isfinite(value):
        return 0.5
    return round(min(1.0, max(0.0, value)), 6) + 0.0


def checkpoint_price(prices: list[float]) -> float:
    """The last price, kept inside [0, 1] and finite for a safe fallback."""
    if not prices:
        return 0.5
    value = float(prices[-1])
    return value if math.isfinite(value) else 0.5


def price_sigma(prices: list[float]) -> float:
    """The per-square-root-minute spread from the market's recent changes."""
    n = len(prices)
    if n < 2:
        return SIGMA_FLOOR
    width = min(MAX_CHANGES, n - 1)
    first = n - 1 - width
    total = 0.0
    for i in range(first, n - 1):
        change = float(prices[i + 1]) - float(prices[i])
        if math.isfinite(change):
            total += change * change
    rms = math.sqrt(total / width)
    level = checkpoint_price(prices)
    floor = max(SIGMA_FLOOR, PRICE_SCALE * math.sqrt(max(0.0, level * (1.0 - level))))
    return min(SIGMA_CAP, max(floor, RMS_SCALE * rms))


def forecast_market(prices: list[float], horizon: int) -> list[list[float]]:
    """One market's rows: row h-1 holds the five quantiles for minute at+h."""
    x = checkpoint_price(prices)
    sigma = price_sigma(prices)
    rows: list[list[float]] = []
    for h in range(1, horizon + 1):
        scale = math.sqrt(h) * sigma
        row = [clamp6(x + z * scale) for z in Z]
        # Rounding cannot break the ordering, but enforce it so every row is valid.
        for j in range(1, len(row)):
            row[j] = max(row[j], row[j - 1])
        rows.append(row)
    return rows


def load_case(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_forecast(case: dict, out_path: Path) -> None:
    horizon = max(0, int(case.get("horizon_minutes", HORIZON_MINUTES)))
    forecasts = []
    for market in case["markets"]:
        rows = forecast_market(market.get("prices", []), horizon)
        if len(rows) != horizon:
            raise ValueError(f"market {market['id']}: wanted {horizon} rows")
        forecasts.append({"id": market["id"], "q": rows})
    text = json.dumps({"forecasts": forecasts}, separators=(",", ":"), allow_nan=False)
    out_path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in"), key=lambda p: p.name):
        write_forecast(load_case(path), OUTPUT_DIR / (path.stem + ".out"))


if __name__ == "__main__":
    main()