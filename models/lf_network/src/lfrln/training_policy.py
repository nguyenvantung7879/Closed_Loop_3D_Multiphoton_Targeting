"""Shared experimental checkpoint selection and equal-specimen aggregation."""
from collections import defaultdict
import math
import numpy as np


def checkpoint_score(metrics, source_metrics):
    return metrics['paired'] / max(source_metrics['paired'], 1e-12)


def checkpoint_eligible(metrics, source_metrics, relative_guard=.05):
    guards = {key: math.isfinite(metrics[key]) and
              metrics[key] <= source_metrics[key] * (1 + relative_guard)
              for key in ('synthetic_clean', 'synthetic_noisy')}
    guards['paired_improved'] = math.isfinite(metrics['paired']) and metrics['paired'] < source_metrics['paired']
    return all(guards.values()), guards


def specimen_average(acquisitions, acquisition_specimens):
    """Mean of frames, then acquisitions, then specimens (equal specimen weights)."""
    by_specimen = defaultdict(list)
    for acquisition, values in acquisitions.items():
        by_specimen[acquisition_specimens[acquisition]].append(float(np.mean(values)))
    means = {key: float(np.mean(values)) for key, values in by_specimen.items()}
    if not means:
        raise ValueError('Empty paired evaluation')
    return float(np.mean(list(means.values()))), means
