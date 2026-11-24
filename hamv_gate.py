"""Standalone numerical checks for HAMV-CC bounded gating (not full HAMV-CC training).

Equations in the revised manuscript:
 alpha_im = mask_im * exp(kappa * (2*r_im-1) - row_max) / normalizer
 b_i = concatenate_m sqrt(M_i * alpha_im) e_im, zero block if unavailable.

Dependencies: numpy. The original estimator / encoders are NOT included here.
"""
from __future__ import annotations
import math
from typing import Sequence
import numpy as np


def strength_for_floor(rho: float, max_available_views: int) -> float:
    """Largest nonnegative kappa guaranteeing share>=rho when M<=max_available_views."""
    if not isinstance(max_available_views, int) or max_available_views < 2:
        raise ValueError("max_available_views must be an integer >=2")
    if not (0 < rho <= 1.0 / max_available_views):
        raise ValueError("rho must be in (0, 1 / max_available_views]")
    return 0.5 * math.log((1.0 / rho - 1.0) / (max_available_views - 1))


def masked_gate(reliability: np.ndarray, mask: np.ndarray, kappa: float) -> np.ndarray:
    """Return normalized masked weights, rejecting all-missing rows.

    Only present-view reliability values are interpreted; absent positions may be NaN.
    """
    r = np.asarray(reliability, dtype=np.float64)
    a = np.asarray(mask, dtype=bool)
    if r.ndim != 2 or r.shape != a.shape:
        raise ValueError("reliability and mask must have equal two-dimensional shape")
    if r.shape[0] == 0 or r.shape[1] == 0:
        raise ValueError("arrays must be non-empty")
    if np.any(~a.any(axis=1)):
        raise ValueError("all-missing document; no silent repair")
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    if not np.all(np.isfinite(r[a])) or np.any((r[a] < 0) | (r[a] > 1)):
        raise ValueError("present-view reliability must be finite within [0,1]")
    u = np.where(a, kappa * (2 * np.where(a, r, 0.5) - 1), -np.inf)
    c = np.max(u, axis=1, keepdims=True)
    x = np.where(a, np.exp(u-c), 0.0)
    return x / x.sum(axis=1, keepdims=True)


def block_fuse(embeddings: Sequence[np.ndarray], mask: np.ndarray,
               alpha: np.ndarray) -> np.ndarray:
    """Preserve each view's coordinate system as a separate, gated block."""
    a = np.asarray(mask, dtype=bool)
    w = np.asarray(alpha, dtype=np.float64)
    if a.ndim != 2 or a.shape != w.shape or len(embeddings) != a.shape[1]:
        raise ValueError("mask/weights/views mismatch")
    if np.any(~a.any(axis=1)) or not np.all(np.isfinite(w)):
        raise ValueError("missing or nonfinite weights")
    if np.any(w < -1e-14) or np.any(np.abs(w[~a]) > 1e-14):
        raise ValueError("nonzero absent-view weight")
    if not np.allclose(w.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError("available weights must sum to one")
    m = a.sum(axis=1)
    blocks = []
    for j, em in enumerate(embeddings):
        e = np.asarray(em, dtype=np.float64)
        if e.ndim != 2 or e.shape[0] != a.shape[0] or not np.all(np.isfinite(e[a[:,j]])):
            raise ValueError("invalid available view embedding")
        scales = np.sqrt(np.maximum(0, m*w[:, j]))
        blocks.append(np.where(a[:, j, None], scales[:, None] * e, 0.0))
    return np.concatenate(blocks, axis=1)


def block_error_bound(n_available: int, max_norm: float, rho: float,
                      kappa: float, delta: float) -> float:
    """Legacy floor-dependent conservative inequality (retained for old audits)."""
    if n_available < 1 or max_norm < 0 or rho <= 0 or kappa < 0 or delta < 0:
        raise ValueError("invalid bound argument")
    return max_norm * np.sqrt(n_available) / (2*np.sqrt(rho)) * min(2., 2*kappa*delta)


def block_error_bound_floorfree(n_available: int, max_norm: float,
                                kappa: float, delta: float) -> float:
    """Revised Proposition 4: global fixed-mask bound with no share-floor premise.

    For b=[sqrt(M*alpha_m)e_m], ||e_m|| <= max_norm and
    ||r-r'||_infty <= delta, the change is bounded by
    R*sqrt(M)*min(kappa*delta, sqrt(2)).
    Does NOT bound changed masks, encoders, or refitted centroids.
    """
    if (not isinstance(n_available, (int, np.integer)) or n_available < 1
            or not np.isfinite(max_norm) or max_norm < 0
            or not np.isfinite(kappa) or kappa < 0
            or not np.isfinite(delta) or delta < 0):
        raise ValueError("invalid bound argument")
    return float(max_norm * np.sqrt(n_available) * min(kappa*delta, np.sqrt(2.0)))


def orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n,n)))
    return q @ np.diag(np.where(np.diag(r)>=0, 1., -1.))


def fixed_centroid_margin_certificate(point: np.ndarray, centroids: np.ndarray,
                                      radius: float) -> tuple[int, float, bool]:
    """Sufficient label-preservation condition under fixed-centroid perturbation.

    No claim applies if the centroids are refitted, the mask changes, or the
    representation/metric differs. A positive certificate requires margin>2*r.
    """
    b=np.asarray(point,float).reshape(-1)
    c=np.asarray(centroids,float)
    if c.ndim!=2 or c.shape[0]<2 or c.shape[1]!=len(b):
        raise ValueError('centroids must contain at least two matching vectors')
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(b)) or not np.isfinite(radius) or radius<0:
        raise ValueError('finite data and nonnegative radius required')
    d=np.linalg.norm(c-b[None,:],axis=1)
    idx=int(np.argmin(d));runner=float(np.partition(d,1)[1]);margin=runner-float(d[idx])
    return idx,margin,bool(margin>2*radius)
