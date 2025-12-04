from typing import Tuple
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors


def downscale_points(points_xy: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """
    Scale 2D point coordinates by (sx, sy).
    """
    if points_xy.size == 0:
        return points_xy
    P = points_xy.astype(np.float32).copy()
    P[:, 0] *= sx
    P[:, 1] *= sy
    return P


def density_from_points_fixed(points_xy: np.ndarray, h: int, w: int, sigma: float = 15.0, normalize_to_count: bool = True) -> np.ndarray:
    dm = np.zeros((h, w), dtype=np.float32)
    if points_xy.size == 0:
        return dm

    xs = np.clip(points_xy[:, 0].astype(int), 0, w - 1)
    ys = np.clip(points_xy[:, 1].astype(int), 0, h - 1)
    dm[ys, xs] = 1.0
    dm = gaussian_filter(dm, sigma=sigma, mode="constant")

    if normalize_to_count:
        s = dm.sum()
        if s > 1e-6:
            dm *= (len(xs) / s)
    return dm


def density_from_points_knn(points_xy: np.ndarray, h: int, w: int, k: int = 3, beta: float = 0.3, normalize_to_count: bool = True,) -> np.ndarray:
    dm = np.zeros((h, w), dtype=np.float32)
    if points_xy.size == 0:
        return dm
    P = points_xy.astype(np.float32)
    P[:, 0] = np.clip(P[:, 0], 0, w - 1)
    P[:, 1] = np.clip(P[:, 1], 0, h - 1)

    n = len(P)
    if n > 1:
        nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm="kd_tree").fit(P)
        dists, _ = nbrs.kneighbors(P)
        if dists.shape[1] > 1:
            sigmas = beta * dists[:, 1 : 1 + min(k, n - 1)].mean(axis=1)
        else:
            sigmas = np.full((n,), 1.5, dtype=np.float32)
    else:
        sigmas = np.full((n,), 1.5, dtype=np.float32)

    for (x, y), s in zip(P, sigmas):
        xi, yi = int(x), int(y)
        rad = max(1, int(3.0 * max(s, 1.0)))
        x0, x1 = max(0, xi - rad), min(w - 1, xi + rad)
        y0, y1 = max(0, yi - rad), min(h - 1, yi + rad)

        xs = np.arange(x0, x1 + 1)
        ys = np.arange(y0, y1 + 1)
        X, Y = np.meshgrid(xs, ys)
        g = np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2.0 * (s ** 2 + 1e-6)))
        dm[y0 : y1 + 1, x0 : x1 + 1] += g.astype(np.float32)

    if normalize_to_count:
        ssum = dm.sum()
        if ssum > 1e-6:
            dm *= (n / ssum)
    return dm