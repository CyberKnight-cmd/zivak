"""
scripts/utils/lr_calculator.py
────────────────────────────────────────────────────────────────────────────────
Pure-Python utility: frequency annotation → sensitivity → LR+ / LR−.

All functions are stateless and side-effect free so they can be imported
by the test suite without touching the filesystem.

Design notes
────────────────────────────────────────────────────────────────────────────────
Sensitivity (Se)
    = P(symptom present | disease present)
    Derived from the HPOA ``frequency_raw`` field.

Background rate (Bg)
    = P(symptom present | disease absent)
    Approximated as:
        Bg[hp_id] = (number of distinct diseases annotated with hp_id)
                    ─────────────────────────────────────────────────
                    (total distinct diseases in the annotation set)
    This is a population-level estimate of how "specific" a symptom is.
    A symptom seen in 800/8000 diseases has Bg = 0.10 — it is a weak
    discriminator.  A symptom seen in 3/8000 has Bg = 0.000375 — it is
    highly specific.

LR+ = Se / Bg
LR− = (1 − Se) / (1 − Bg)

Clamping
    Sensitivities are clamped to [SE_MIN, SE_MAX] = [0.005, 0.995] before
    any LR calculation to avoid division-by-zero and degenerate infinities
    while still producing extreme (but finite) LR values at the boundaries.
    Background rates are clamped to [BG_MIN, BG_MAX] = [0.001, 0.999] for
    the same reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ── tuneable constants ────────────────────────────────────────────────────────

SE_MIN: float = 0.005   # 0.5 %  — floor for sensitivity (avoids LR+ = 0)
SE_MAX: float = 0.995   # 99.5 % — ceiling for sensitivity (avoids LR- = 0)
BG_MIN: float = 0.001   # 0.1 %  — floor for background rate
BG_MAX: float = 0.999   # 99.9 % — ceiling for background rate

# ── HPO frequency term → point-estimate of sensitivity ───────────────────────
#
#   Source: https://hpo.jax.org/app/browse/term/HP:0040279
#   HP:0040280  Obligate       = 100 %          → 1.00
#   HP:0040281  Very frequent  = 80  – 99 %     → midpoint 0.895
#   HP:0040282  Frequent       = 30  – 79 %     → midpoint 0.545
#   HP:0040283  Occasional     =  5  – 29 %     → midpoint 0.170
#   HP:0040284  Very rare      =  1  –  4 %     → midpoint 0.025
#   HP:0040285  Excluded       =  0  %           → 0.00
#
HP_FREQ_SENSITIVITY: dict[str, float] = {
    "HP:0040280": 1.000,   # Obligate
    "HP:0040281": 0.895,   # Very frequent
    "HP:0040282": 0.545,   # Frequent
    "HP:0040283": 0.170,   # Occasional
    "HP:0040284": 0.025,   # Very rare
    "HP:0040285": 0.000,   # Excluded
}

# Pre-compiled patterns for ratio and percentage parsing
_RATIO_RE   = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
_PERCENT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%\s*$")

# Plain-text frequency labels that appear in HPOA alongside HP: CURIEs.
# Matched case-insensitively after stripping whitespace.
# "excluded" maps to None so the caller skips the row entirely.
_PLAIN_TEXT_FREQ: dict[str, Optional[float]] = {
    "very frequent": 0.90,
    "frequent":      0.50,
    "occasional":    0.17,
    "rare":          0.025,
    "very rare":     0.005,
    "hallmark":      1.00,
    "obligate":      1.00,
    "excluded":      None,
}


# ── public data class ─────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class LRResult:
    """Holds a single computed likelihood-ratio pair."""
    sensitivity:  float
    lr_positive:  float
    lr_negative:  float


# ── frequency → sensitivity ───────────────────────────────────────────────────

def parse_frequency(raw: str) -> Optional[float]:
    """
    Convert a raw HPOA frequency string to a sensitivity value in [0, 1].

    Accepted formats
    ----------------
    ``"HP:0040281"``   HPO frequency term
    ``"12/34"``        explicit ratio
    ``"45%"``          percentage
    ``""``             empty → ``None``  (caller should skip this row)

    Parameters
    ----------
    raw : str
        The ``frequency_raw`` value from the parsed HPOA annotations.

    Returns
    -------
    float | None
        Sensitivity in the range [0.0, 1.0], or ``None`` when the field
        is empty / unrecognisable (caller decides how to handle).

    Notes
    -----
    * Ratios with a zero denominator return ``None``.
    * Values outside [0, 1] after parsing are clamped and logged implicitly
      (the caller should apply :func:`clamp_sensitivity` before further use).
    """
    s = raw.strip()

    if not s:
        return None

    # HPO frequency term
    if s in HP_FREQ_SENSITIVITY:
        return HP_FREQ_SENSITIVITY[s]

    # Plain-text frequency label (case-insensitive)
    lower = s.lower()
    if lower in _PLAIN_TEXT_FREQ:
        return _PLAIN_TEXT_FREQ[lower]

    # Ratio  e.g.  "12/34"
    m = _RATIO_RE.match(s)
    if m:
        numerator, denominator = int(m.group(1)), int(m.group(2))
        if denominator == 0:
            return None
        return numerator / denominator

    # Percentage  e.g.  "45%"
    m = _PERCENT_RE.match(s)
    if m:
        return float(m.group(1)) / 100.0

    # Unrecognised — caller receives None and will skip
    return None


def clamp_sensitivity(se: float) -> float:
    """Clamp sensitivity to [SE_MIN, SE_MAX] before LR computation."""
    return max(SE_MIN, min(SE_MAX, se))


def clamp_background(bg: float) -> float:
    """Clamp background rate to [BG_MIN, BG_MAX] before LR computation."""
    return max(BG_MIN, min(BG_MAX, bg))


# ── core LR calculation ───────────────────────────────────────────────────────

def compute_lr(sensitivity: float, background_rate: float) -> LRResult:
    """
    Compute LR+ and LR− from sensitivity and background rate.

    Both inputs are clamped internally — pass raw values, not pre-clamped.

    Parameters
    ----------
    sensitivity : float
        P(symptom | disease) in [0, 1].
    background_rate : float
        P(symptom | ¬disease) in [0, 1].
        Pass the pre-computed per-HP-term population background rate.

    Returns
    -------
    LRResult
        Named tuple with ``sensitivity``, ``lr_positive``, ``lr_negative``.

    Formula
    -------
    ::

        LR+ = Se  /  Bg
        LR− = (1 − Se) / (1 − Bg)

    Examples
    --------
    >>> r = compute_lr(0.895, 0.10)
    >>> round(r.lr_positive, 2)
    8.95
    >>> round(r.lr_negative, 2)
    0.12
    """
    se = clamp_sensitivity(sensitivity)
    bg = clamp_background(background_rate)

    lr_pos = se / bg
    lr_neg = (1.0 - se) / (1.0 - bg)

    return LRResult(
        sensitivity=round(se, 6),
        lr_positive=round(lr_pos, 6),
        lr_negative=round(lr_neg, 6),
    )
