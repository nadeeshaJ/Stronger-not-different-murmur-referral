"""Platt calibration and the 90%-sensitivity threshold, both fitted on pooled inner out-of-fold scores."""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


class Platt:
    """Unpenalised logistic regression of y on logit(score): p = sigmoid(a * logit(s) + b)."""

    def __init__(self):
        self.a, self.b = 1.0, 0.0

    def fit(self, score, y, max_iter: int = 100, tol: float = 1e-10):
        """Newton-Raphson (IRLS) on the 2-parameter logistic likelihood; exact MLE, no penalty."""
        z, y = logit(score), np.asarray(y, float)
        X = np.column_stack([z, np.ones_like(z)])
        w = np.array([1.0, 0.0])
        for _ in range(max_iter):
            mu = sigmoid(X @ w)
            g = X.T @ (y - mu)
            H = X.T @ (X * (mu * (1 - mu))[:, None])
            step = np.linalg.solve(H, g)
            # step-halving keeps the likelihood non-decreasing (guards against separation blow-ups)
            ll0 = np.sum(y * (X @ w) - np.logaddexp(0, X @ w))
            t = 1.0
            while t > 1e-6:
                wn = w + t * step
                if np.sum(y * (X @ wn) - np.logaddexp(0, X @ wn)) >= ll0 - 1e-12:
                    break
                t /= 2
            w = wn
            if np.max(np.abs(t * step)) < tol:
                break
        else:
            raise RuntimeError("Platt fit did not converge")
        self.a, self.b = map(float, w)
        return self

    def transform(self, score):
        return sigmoid(self.a * logit(score) + self.b)

    def to_dict(self):
        return {"a": self.a, "b": self.b}


def sensitivity_threshold(p, y, target: float = 0.90) -> float:
    """Largest threshold t such that sensitivity(p >= t) >= target on the given (development) data.
    Choosing the largest such t maximises specificity subject to the sensitivity constraint."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    pos = np.sort(p[y == 1])[::-1]
    n = len(pos)
    if n == 0:
        raise ValueError("no positives")
    k = int(np.ceil(target * n - 1e-12))      # need at least k positives at or above t
    return float(pos[k - 1])


def calibration_slope_intercept(p, y):
    """Calibration slope and intercept (logistic recalibration of y on logit(p))."""
    m = Platt().fit(p, y)
    return m.a, m.b
