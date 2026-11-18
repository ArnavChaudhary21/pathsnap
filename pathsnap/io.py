"""Raw-file readers for the PathSnap dataset.

Each trip lives in its own folder ``<partition>/<trip_id>/`` holding
``<trip_id>.track``, ``.nodes``, ``.arcs`` and (optionally) ``.route``.

Prediction code must only ever call :func:`load_trip_inputs`. The reference
route is read through the separate :func:`load_reference_route` so that the
ground truth never travels with the inputs of the prediction pipeline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Student folder", "Student folder",
)


def data_root() -> str:
    return os.environ.get("PATHSNAP_DATA", DEFAULT_DATA_ROOT)


def list_trips(partition: str, root: str | None = None) -> list[str]:
    base = os.path.join(root or data_root(), partition)
    return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))


def trip_dir(partition: str, trip_id: str, root: str | None = None) -> str:
    return os.path.join(root or data_root(), partition, trip_id)


@dataclass
class TripInputs:
    trip_id: str
    track: np.ndarray   # (n, 3) lon, lat, t
    nodes: np.ndarray   # (N, 2) lon, lat
    arcs: np.ndarray    # (A, 2) from_node, to_node


def _read(path: str, dtype, ncols: int) -> np.ndarray:
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if arr.size == 0:
        return np.zeros((0, ncols), dtype=dtype)
    if arr.shape[1] != ncols:
        raise ValueError(f"{path}: expected {ncols} columns, got {arr.shape[1]}")
    return arr


def load_trip_inputs(folder: str) -> TripInputs:
    tid = os.path.basename(os.path.normpath(folder))
    p = os.path.join(folder, tid)
    return TripInputs(
        trip_id=tid,
        track=_read(p + ".track", float, 3),
        nodes=_read(p + ".nodes", float, 2),
        arcs=_read(p + ".arcs", np.int64, 2),
    )


def load_reference_route(folder: str) -> np.ndarray | None:
    """Ground-truth arc sequence. Only for label building and evaluation."""
    tid = os.path.basename(os.path.normpath(folder))
    p = os.path.join(folder, tid + ".route")
    if not os.path.exists(p):
        return None
    return np.loadtxt(p, dtype=np.int64, ndmin=1)
