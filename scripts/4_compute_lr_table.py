#!/usr/bin/env python3
"""
scripts/4_compute_lr_table.py
--------------------------------------------------------------------------------
Computes lr_table.csv using a prevalence-weighted background rate formula:

    background_rate(hp_id) = sum(prevalence_i * sensitivity_i)
                             for all diseases i annotated with hp_id

All joins are vectorized via .map() and groupby.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils.lr_calculator import parse_frequency, BG_MIN, BG_MAX, SE_MIN, SE_MAX  # noqa: E402

BASE_DIR   = SCRIPT_DIR.parent
DATA_DIR   = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
PROCESSED  = DATA_DIR / "processed"

HPOA_FILE       = PROCESSED / "hpoa_annotations.json"
DISEASE_INDEX   = PROCESSED / "disease_index.json"
SYMPTOM_INDEX   = PROCESSED / "symptom_index.json"
OMIM_TO_DOID    = PROCESSED / "omim_to_doid.json"
ORPHA_TO_DOID   = PROCESSED / "orpha_to_doid.json"
OUTPUT          = PROCESSED / "lr_table.csv"

LR_MIN = 0.01
LR_MAX = 100.0


def main() -> None:
    for path in (HPOA_FILE, DISEASE_INDEX, SYMPTOM_INDEX, OMIM_TO_DOID, ORPHA_TO_DOID):
        if not path.exists():
            print(f"ERROR: required file not found: {path}", file=sys.stderr)
            sys.exit(1)

    print("Loading inputs ...")
    hpoa_raw      = json.loads(HPOA_FILE.read_text(encoding="utf-8"))
    disease_raw   = json.loads(DISEASE_INDEX.read_text(encoding="utf-8"))
    symptom_raw   = json.loads(SYMPTOM_INDEX.read_text(encoding="utf-8"))
    omim_to_doid  = json.loads(OMIM_TO_DOID.read_text(encoding="utf-8"))
    orpha_to_doid = json.loads(ORPHA_TO_DOID.read_text(encoding="utf-8"))

    df = pd.DataFrame(hpoa_raw)
    print(f"  Annotations loaded: {len(df):,}")

    combined_bridge: dict[str, str] = {**omim_to_doid, **orpha_to_doid}
    df["disease_doid"] = df["disease_id"].map(combined_bridge)

    before = len(df)
    df = df.dropna(subset=["disease_doid"])
    print(f"  After DOID resolution: {len(df):,} rows (dropped {before - len(df):,} unmappable)")

    doid_to_prevalence   = {doid: info["prevalence"] for doid, info in disease_raw.items()}
    doid_to_disease_name = {doid: info["name"]       for doid, info in disease_raw.items()}

    df["prevalence"]   = df["disease_doid"].map(doid_to_prevalence)
    df["disease_name"] = df["disease_doid"].map(doid_to_disease_name)
    df["disease_omim"] = df["disease_id"]

    before = len(df)
    df = df.dropna(subset=["prevalence"])
    if len(df) < before:
        print(f"  Dropped {before - len(df):,} rows with unknown DOID in disease_index")

    df["sensitivity"] = df["frequency_raw"].map(parse_frequency)

    before = len(df)
    df = df.dropna(subset=["sensitivity"])
    print(f"  After sensitivity parsing: {len(df):,} rows (dropped {before - len(df):,} unrecognised frequency)")

    df["sensitivity"] = df["sensitivity"].clip(lower=SE_MIN, upper=SE_MAX)

    df["prev_x_se"]       = df["prevalence"] * df["sensitivity"]
    df["background_rate"] = df.groupby("hpo_id")["prev_x_se"].transform("sum")
    df["background_rate"] = df["background_rate"].clip(lower=BG_MIN, upper=BG_MAX)

    df["lr_positive"] = df["sensitivity"] / df["background_rate"]
    df["lr_negative"] = (1.0 - df["sensitivity"]) / (1.0 - df["background_rate"])

    df["lr_positive"] = df["lr_positive"].clip(lower=LR_MIN, upper=LR_MAX)
    df["lr_negative"] = df["lr_negative"].clip(lower=LR_MIN, upper=LR_MAX)

    hp_to_name = {hp_id: info["name"] for hp_id, info in symptom_raw.items()}
    df["symptom_name"] = df["hpo_id"].map(hp_to_name)

    out = df[["disease_omim", "disease_doid", "disease_name", "hpo_id",
              "symptom_name", "sensitivity", "lr_positive", "lr_negative",
              "frequency_raw"]].copy()
    out = out.rename(columns={"hpo_id": "hp_id"})
    out.to_csv(OUTPUT, index=False)

    n_rows       = len(out)
    lr_pos_stats = out["lr_positive"]
    lr_neg_stats = out["lr_negative"]
    print(f"\nRow count: {n_rows:,}")
    print(f"lr_positive  min={lr_pos_stats.min():.4f}  max={lr_pos_stats.max():.2f}"
          f"  mean={lr_pos_stats.mean():.2f}  median={lr_pos_stats.median():.2f}")
    print(f"lr_negative  min={lr_neg_stats.min():.4f}  max={lr_neg_stats.max():.2f}"
          f"  mean={lr_neg_stats.mean():.2f}  median={lr_neg_stats.median():.2f}")
    print(f"Rows where lr_positive hit 100 cap:  {(lr_pos_stats >= LR_MAX).sum():,}")
    print(f"Rows where lr_positive hit 0.01 cap: {(lr_pos_stats <= LR_MIN).sum():,}")
    print(f"Output written to: {OUTPUT}")


if __name__ == "__main__":
    main()
