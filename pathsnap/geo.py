"""Coordinate handling.

Trips are spread across the globe (longitudes from ~2 to ~136 degrees), so a
single projection does not work. Each trip is projected to the UTM zone of its
GPS centroid; trip extents are a few km to a few tens of km, where UTM scale
error is < 0.1 %, which is far below GPS noise.
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.epsg = utm_epsg(lon0, lat0)
        self._fwd = Transformer.from_crs(4326, self.epsg, always_xy=True)
        self._inv = Transformer.from_crs(self.epsg, 4326, always_xy=True)

    @classmethod
    def for_track(cls, track_lonlat: np.ndarray) -> "LocalProjection":
        return cls(float(np.median(track_lonlat[:, 0])), float(np.median(track_lonlat[:, 1])))

    def to_xy(self, lonlat: np.ndarray) -> np.ndarray:
        x, y = self._fwd.transform(lonlat[:, 0], lonlat[:, 1])
        return np.column_stack([x, y])

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        lon, lat = self._inv.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([lon, lat])


def haversine_m(lon1, lat1, lon2, lat2):
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def point_segment(px, py, ax, ay, bx, by):
    """Vectorised point-to-segment projection.

    Returns (distance, t) where t in [0, 1] is the clamped fraction along a->b.
    """
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * dx + (py - ay) * dy) / L2
    t = np.where(L2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    qx, qy = ax + t * dx, ay + t * dy
    return np.hypot(px - qx, py - qy), t


def angle_diff(a, b):
    """Absolute smallest difference between two bearings in radians, in [0, pi]."""
    d = np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)
    return d
