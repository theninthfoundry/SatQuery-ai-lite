"""
backend/uncertainty/stability.py

Which changed pixels survive every tested threshold?

For the monotone rule "changed if |A-B| > t":
  stable    = |A-B| > t_max     (changed at EVERY tested threshold)
  ambiguous = t_min < |A-B| <= t_max  (changed at some thresholds only)

Stability is robustness to the *threshold choice*. It is not evidence that a
change is real: a uniform exposure shift is perfectly stable.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def stability_masks(diff: np.ndarray, t_min: int, t_max: int) -> Tuple[np.ndarray, np.ndarray]:
    stable = diff > t_max
    ambiguous = (diff > t_min) & ~stable
    return stable, ambiguous
