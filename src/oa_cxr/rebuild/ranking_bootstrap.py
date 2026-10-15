"""Exact frequency-weighted ranking statistics with label-blind score ties.

Integer weights represent replicated observations in a cluster bootstrap.
Harmonic sums evaluate the existing discrete AURC without expanding repeats.
"""
from __future__ import annotations
import numpy as np

RANKING_METRICS = ('auroc', 'auprc', 'aurc', 'excess_aurc')
COVERAGES = np.array([.5, .6, .7, .8, .9, 1.])


class RankingCache:
    def __init__(self, truth, scores, groups, group_count):
        truth, scores = np.asarray(truth, dtype=float), np.asarray(scores, dtype=float)
        groups = np.asarray(groups)
        if truth.ndim != 1 or scores.shape != truth.shape or groups.shape != truth.shape or not len(truth):
            raise ValueError('Nonempty aligned ranking vectors required')
        if not np.isfinite(truth).all() or not np.isfinite(scores).all() or np.any((truth < 0) | (truth > 1)):
            raise ValueError('Invalid truth or scores')
        if groups.dtype.kind not in 'iu' or groups.min() < 0 or groups.max() >= group_count:
            raise ValueError('Invalid bootstrap groups')
        order = np.argsort(-scores, kind='stable')
        sorted_scores = scores[order]
        self.groups = groups[order]
        self.bad = (truth[order] < .9).astype(np.int64)
        self.starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
        self.group_count = group_count
        self.group_max_rows = int(np.bincount(groups, minlength=group_count).max())
        self.harmonic = np.r_[0., np.cumsum(1. / np.arange(1, max(1, group_count * self.group_max_rows) + 1))]

    def blocks(self, weights):
        weights = np.asarray(weights)
        if weights.shape != (self.group_count,) or not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights != weights.astype(np.int64)):
            raise ValueError('Nonnegative integer group multiplicities required')
        row_weights = weights.astype(np.int64)[self.groups]
        total = np.add.reduceat(row_weights, self.starts)
        bad = np.add.reduceat(row_weights * self.bad, self.starts)
        keep = total > 0
        return total[keep], bad[keep]

    def evaluate(self, weights, coverages=COVERAGES):
        total, bad = self.blocks(weights)
        n, b = int(total.sum()), int(bad.sum())
        if not n: return np.full(4 + len(coverages), np.nan)
        g = n - b
        if n >= len(self.harmonic):
            self.harmonic = np.r_[0., np.cumsum(1. / np.arange(1, n + 1))]
        ends = np.cumsum(total); starts = ends - total
        before = np.cumsum(bad) - bad
        fraction = bad / total
        aurc = float(np.sum(bad + (before - starts * fraction) *
                            (self.harmonic[ends] - self.harmonic[starts])) / n)
        oracle = float((b - g * (self.harmonic[n] - self.harmonic[g])) / n)
        if b and g:
            good = total - bad
            auroc = float(np.sum(bad * (np.cumsum(good) - good / 2)) / (b * g))
            auprc = float(np.sum(bad[::-1] * np.cumsum(bad[::-1]) / np.cumsum(total[::-1])) / b)
        else:
            auroc, auprc = np.nan, np.nan
        coverages = np.asarray(coverages)
        if np.any((coverages <= 0) | (coverages > 1)): raise ValueError('Coverage must be in (0, 1]')
        kept = np.ceil(n * coverages).astype(np.int64)
        idx = np.searchsorted(ends, kept, side='left')
        risk = (before[idx] + (kept - starts[idx]) * fraction[idx]) / kept
        return np.r_[auroc, auprc, aurc, aurc - oracle, risk]

    def curves(self, weights, resolution=201):
        total, bad = self.blocks(weights)
        n, b = int(total.sum()), int(bad.sum()); g = n - b
        good = total - bad
        recall = np.r_[0., np.cumsum(bad[::-1]) / b] if b else np.array([])
        fpr = np.r_[0., np.cumsum(good[::-1]) / g] if g else np.array([])
        precision = np.r_[1., np.cumsum(bad[::-1]) / np.cumsum(total[::-1])] if b else np.array([])
        coverage = np.linspace(1 / n, 1., resolution)
        risk = self.evaluate(weights, coverage)[4:]
        return dict(fpr=fpr, tpr=recall, recall=recall, precision=precision,
                    coverage=coverage, risk=risk)
