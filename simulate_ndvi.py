#!/usr/bin/env python3
"""Simulate NDVI values using spectral filter profiles.

This script demonstrates how to combine a filter transmission curve with
reflectance spectra to compute NDVI. It expects the filter profile in the
same JSON format as ``SERAPH_R118_absorbtion (1).json`` and a CSV file with
reflectance spectra columns labeled by wavelength in nanometers.

Example usage::

    python simulate_ndvi.py --red-filter 'SERAPH_R118_absorbtion (1).json' \
        --csv intact_spec.csv

If ``--nir-filter`` is omitted, a simple box filter of 780--900 nm is used.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from typing import Iterable, List, Tuple


def load_filter_json(path: str) -> Tuple[List[float], List[float]]:
    """Return (wavelengths, transmissions) from a JSON filter description."""
    with open(path) as f:
        data = json.load(f)
    pairs = data["datasetColl"][0]["data"]
    # Some files contain a trailing sample that should be dropped
    wl = [p["value"][0] for p in pairs[:-1]]
    tr = [p["value"][1] for p in pairs[:-1]]
    return wl, tr


def linear_interp(x: float, xs: List[float], ys: List[float]) -> float:
    """Simple linear interpolation."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_left(xs, x)
    x0, x1 = xs[i - 1], xs[i]
    y0, y1 = ys[i - 1], ys[i]
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def band_value(
    wl: Iterable[float],
    refl: Iterable[float],
    filter_wl: List[float],
    filter_tr: List[float],
) -> float:
    """Weighted average reflectance using a spectral filter."""
    num = 0.0
    denom = 0.0
    for w, r in zip(wl, refl):
        f = linear_interp(w, filter_wl, filter_tr)
        num += r * f
        denom += f
    return num / denom if denom else math.nan


def compute_ndvi(
    wavelengths: List[float],
    reflectances: List[float],
    red_filter: Tuple[List[float], List[float]],
    nir_filter: Tuple[List[float], List[float]],
) -> float:
    red = band_value(wavelengths, reflectances, *red_filter)
    nir = band_value(wavelengths, reflectances, *nir_filter)
    if red + nir == 0:
        return math.nan
    return (nir - red) / (nir + red)


def box_filter(start: float, end: float) -> Tuple[List[float], List[float]]:
    """Return a simple box filter (flat response)."""
    return [start, end], [1.0, 1.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate NDVI with spectral filters")
    p.add_argument("--csv", default="intact_spec.csv", help="CSV with reflectance data")
    p.add_argument("--red-filter", default="SERAPH_R118_absorbtion (1).json")
    p.add_argument(
        "--nir-filter",
        help="JSON file for NIR filter (default: 780-900 nm box filter)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    red_filter = load_filter_json(args.red_filter)
    if args.nir_filter and args.nir_filter.endswith(".json"):
        nir_filter = load_filter_json(args.nir_filter)
    else:
        nir_filter = box_filter(780, 900)

    with open(args.csv) as f:
        reader = csv.DictReader(f)
        wl_fields = [int(c) for c in reader.fieldnames if c.isdigit()]
        wl_fields.sort()
        ndvi_values = []
        for row in reader:
            refl = [float(row[str(w)]) for w in wl_fields]
            ndvi = compute_ndvi(wl_fields, refl, red_filter, nir_filter)
            ndvi_values.append(ndvi)

    if ndvi_values:
        mn = min(ndvi_values)
        mx = max(ndvi_values)
        print(f"Computed {len(ndvi_values)} NDVI values")
        print(f"Range: {mn:.3f} – {mx:.3f}")
    else:
        print("No data rows found in CSV")


if __name__ == "__main__":
    main()
