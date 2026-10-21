"""Explicit epoch-based noise curriculum without changing optical input units."""
import math
import torch
from .datasets import MixedLFNoiseDataset


def parse_noise_schedule(spec):
    if not spec:
        return ()
    stages = []
    for token in spec.split(','):
        epoch, probability = token.strip().split(':')
        epoch, probability = int(epoch), float(probability)
        if epoch < 1 or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Noise schedule requires positive epochs and probabilities in [0,1]')
        if stages and epoch <= stages[-1][0]:
            raise ValueError('Noise schedule epochs must increase strictly')
        stages.append((epoch, probability))
    if stages[0][0] != 1:
        raise ValueError('Noise schedule must begin at epoch 1')
    return tuple(stages)


def noise_probability_at_epoch(stages, epoch):
    if epoch < 1 or not stages:
        raise ValueError('A schedule and positive epoch are required')
    return next(p for start, p in reversed(stages) if epoch >= start)


def set_training_noise_epoch(loader, args, epoch):
    stages = parse_noise_schedule(getattr(args, 'noise_schedule', None))
    if not stages:
        return float(getattr(loader.dataset, 'noise_probability', 0.))
    if not isinstance(loader.dataset, MixedLFNoiseDataset):
        raise TypeError('Scheduled training needs a MixedLFNoiseDataset wrapper')
    seed = getattr(args, 'training_seed', 20260926)
    # Separate shuffle/geometric-transform RNG from sensor draws. A given sample
    # gets a reproducible fresh draw each epoch, matched across noise policies.
    torch.manual_seed(seed + epoch)
    if loader.generator is not None:
        loader.generator.manual_seed(seed + epoch)
    loader.dataset.noise_probability = noise_probability_at_epoch(stages, epoch)
    loader.dataset.deterministic = True
    loader.dataset.seed = seed + epoch * 1000003
    return loader.dataset.noise_probability


def phase_start_epoch(checkpoint_epoch, *, continuing):
    """New phases reset their epoch; exact continuation retains progress."""
    return int(checkpoint_epoch) if continuing else 0
