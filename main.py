#!/usr/bin/env python3
"""Forecast the next hours of Polymarket's "Next CEO of NVIDIA?" Yes prices.

Reads every <id>.in case from $MAKEFUN_INPUT_DIR (default /input) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (default /output). For each market it forecasts
a random-walk spread around the checkpoint price, with the per-minute spread
estimated from the last 60 one-minute changes:

    sigma = min(0.005, max(0.0002, 0.45 * rms(d), 0.0035 * sqrt(p * (1 - p))))
    drift = (1 - S) * x * sqrt(h / 240)
    q_tau(h) = clamp(x + drift + z_tau * sqrt(h) * sigma, 0, 1)

where S is the sum of every checkpoint price in the case: the Yes prices of one
event should sum to 1, so an overround or underround is expected to revert.
The upper quantiles use a slightly tighter z. Rounded to 6 decimals. Python 3
standard library only, no network, no randomness.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.607041, 1.480369]
MAX_CHANGES = 60
SIGMA_FLOOR = 0.0002
SIGMA_CAP = 0.005
RMS_SCALE = 0.45
PRICE_SCALE = 0.0035
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


def forecast_market(prices: list[float], horizon: int, sum_price: float) -> list[list[float]]:
    """One market's rows: row h-1 holds the five quantiles for minute at+h."""
    x = checkpoint_price(prices)
    sigma = price_sigma(prices)
    pull = 1.0 - sum_price
    rows: list[list[float]] = []
    for h in range(1, horizon + 1):
        scale = math.sqrt(h) * sigma
        drift = pull * x * math.sqrt(h / max(1, horizon))
        row = [clamp6(x + drift + z * scale) for z in Z]
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
    markets = case["markets"]
    priced = [checkpoint_price(m.get("prices", [])) for m in markets if m.get("prices")]
    sum_price = sum(priced) if priced else 1.0
    forecasts = []
    for market in markets:
        rows = forecast_market(market.get("prices", []), horizon, sum_price)
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
