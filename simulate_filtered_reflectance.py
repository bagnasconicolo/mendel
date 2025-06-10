#!/usr/bin/env python3
"""Simulate filtered reflectance for mixtures of vegetation types.

This script mixes leaf litter, healthy leaf, and ground reflectance profiles
in varying percentages and reports the sensor reading after passing through a
spectral filter.  The filter profile should be provided in the same JSON
format as ``SERAPH_R118_absorbtion (1).json``.  Note that this R118 profile
represents a **blue** filter rather than a red one.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_filter_json(path: str) -> Tuple[List[float], List[float]]:
    """Return (wavelengths, transmissions) from a JSON filter description."""
    with open(path) as f:
        data = json.load(f)
    pairs = data["datasetColl"][0]["data"]
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
    return num / denom if denom else float("nan")


def read_average_dictcsv(path: str) -> Tuple[List[int], List[float]]:
    """Return wavelengths and mean reflectance from a DictReader CSV."""
    with open(path) as f:
        reader = csv.DictReader(f)
        wl_fields = [int(c) for c in reader.fieldnames if c.isdigit()]
        wl_fields.sort()
        spectra = []
        for row in reader:
            spectra.append([float(row[str(w)]) for w in wl_fields])
    mean = [sum(col) / len(col) for col in zip(*spectra)]
    return wl_fields, mean


def read_average_rowcsv(path: str) -> Tuple[List[int], List[float]]:
    """Return wavelengths and mean reflectance from a simple row-oriented CSV."""
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        if header and header[0].startswith("\ufeff"):
            header[0] = header[0].lstrip("\ufeff")
        wl = [int(h) for h in header if h]
        spectra = []
        for row in reader:
            if not row:
                continue
            spectra.append([float(x) for x in row[: len(wl)]])
    mean = [sum(col) / len(col) for col in zip(*spectra)]
    return wl, mean


def mix_spectra(
    wl_common: List[int],
    litter: Tuple[List[int], List[float]],
    healthy: Tuple[List[int], List[float]],
    ground: Tuple[List[int], List[float]],
    weights: Tuple[float, float, float],
) -> List[float]:
    """Return mixed reflectance for wl_common with given weights."""
    w_lit, r_lit = litter
    w_h, r_h = healthy
    w_g, r_g = ground

    def interp(w, wl, refl):
        return linear_interp(w, wl, refl)

    result = []
    for w in wl_common:
        val = (
            weights[0] * interp(w, w_lit, r_lit)
            + weights[1] * interp(w, w_h, r_h)
            + weights[2] * interp(w, w_g, r_g)
        )
        result.append(val)
    return result


def plot_heatmap(values: Dict[Tuple[int, int], float], step: int, filter_name: str) -> None:
    """Plot a heatmap of filter-weighted reflectance."""
    steps = list(range(0, 101, step))
    idx = {v: i for i, v in enumerate(steps)}
    matrix = np.full((len(steps), len(steps)), np.nan)
    for (leaf, litter), val in values.items():
        matrix[idx[litter], idx[leaf]] = val
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        matrix,
        origin="lower",
        extent=[0, 100, 0, 100],
        cmap="viridis",
        aspect="auto",
    )
    ax.set_xlabel("Leaf %")
    ax.set_ylabel("Litter %")
    ax.set_title(f"Filtered reflectance ({filter_name})")
    fig.colorbar(im, ax=ax, label="Reflectance")
    plt.tight_layout()
    plt.savefig("filtered_reflectance.png")
    print("Saved plot to filtered_reflectance.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulate filtered reflectance for mixtures of leaf litter, healthy leaf, and ground",
    )
    p.add_argument("--step", type=int, default=25, help="percentage step size")
    p.add_argument(
        "--filter",
        default="SERAPH_R118_absorbtion (1).json",
        help="JSON file describing the spectral filter",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    filt = load_filter_json(args.filter)

    wl_lit, refl_lit = read_average_dictcsv("intact_spec.csv")
    wl_g, refl_g = read_average_dictcsv("ground_spec.csv")
    wl_h, refl_h = read_average_rowcsv(
        "2012-leaf-reflectance-spectra-of-tropical-trees-in-tapajos-national-forest.csv"
    )

    wl_common = sorted(set(wl_lit) & set(wl_g) & set(wl_h))

    step = args.step
    values: Dict[Tuple[int, int], float] = {}
    for leaf_pct in range(0, 101, step):
        for litter_pct in range(0, 101 - leaf_pct, step):
            ground_pct = 100 - leaf_pct - litter_pct
            weights = (
                litter_pct / 100.0,
                leaf_pct / 100.0,
                ground_pct / 100.0,
            )
            mixed = mix_spectra(
                wl_common,
                (wl_lit, refl_lit),
                (wl_h, refl_h),
                (wl_g, refl_g),
                weights,
            )
            value = band_value(wl_common, mixed, *filt)
            print(
                f"litter {litter_pct:3d}%, leaf {leaf_pct:3d}%, ground {ground_pct:3d}% -> reading {value:.3f}"
            )
            values[(leaf_pct, litter_pct)] = value

    plot_heatmap(values, step, args.filter)


if __name__ == "__main__":
    main()
