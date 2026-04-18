"""Calibrated confidence for the face-verification pipeline.

Confidence rule
---------------
The confidence value reflects how certain the system is in its binary
decision (same / different).  It is derived from the **distance between
the similarity score and the operating threshold**, normalised to the
[0, 1] range.

    distance = score - threshold

For a cosine-similarity score the theoretical range is [-1, 1].

* If **score >= threshold** (decision = same):
      confidence = 0.5 + 0.5 * min(1, distance / (1.0 - threshold))

* If **score < threshold** (decision = different):
      confidence = 0.5 + 0.5 * min(1, |distance| / (threshold - (-1.0)))

Interpretation:
    0.5  = borderline (score sits right at the threshold)
    1.0  = maximum certainty (score at the extreme end of its range)

The rule is reproducible, requires no learned parameters, and is fully
determined by the score and threshold.
"""

from __future__ import annotations

import numpy as np


def calibrated_confidence(
    score: float | np.ndarray,
    threshold: float,
    score_min: float = -1.0,
    score_max: float = 1.0,
) -> float | np.ndarray:
    """Compute calibrated confidence for a score (or array of scores).

    Parameters
    ----------
    score : float or array
        Cosine similarity score(s).
    threshold : float
        Operating threshold (decision boundary).
    score_min, score_max : float
        Theoretical bounds of the score range (default: cosine [-1, 1]).

    Returns
    -------
    confidence : same shape as *score*, values in [0.5, 1.0].
    """
    score = np.asarray(score, dtype=float)
    scalar = score.ndim == 0
    score = np.atleast_1d(score)

    distance = score - threshold

    # Range available on each side of the threshold
    range_above = score_max - threshold  # for "same" predictions
    range_below = threshold - score_min  # for "different" predictions

    # Avoid division by zero when threshold sits at an extreme
    range_above = max(range_above, 1e-9)
    range_below = max(range_below, 1e-9)

    conf = np.where(
        distance >= 0,
        0.5 + 0.5 * np.clip(distance / range_above, 0.0, 1.0),
        0.5 + 0.5 * np.clip(np.abs(distance) / range_below, 0.0, 1.0),
    )

    return float(conf[0]) if scalar else conf
