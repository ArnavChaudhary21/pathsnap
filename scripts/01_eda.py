"""Checkpoint 1: dataset exploration (train + student_test partitions only).

Outputs
  results/eda/inventory.csv           per-trip inventory and data-quality stats
  results/eda/summary.json            partition-level summary
  results/eda/*.png                   figures used in the report
  results/maps/eda_trip_<id>.html     Folium map (GPS + network + reference route)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.geo import LocalProjection, haversine_m
from pathsnap.graph import RoadGraph
from pathsnap.io import list_trips, load_reference_route, load_trip_inputs, trip_dir
from pathsnap.labels import align_track_to_route
from pathsnap.viz import trip_map

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "eda")
MAPS = os.path.join(ROOT, "results", "maps")
os.makedirs(OUT, exist_ok=True)
os.makedirs(MAPS, exist_ok=True)


def segs(g, ids, **kw):
    return LineCollection(np.stack([np.column_stack([g.ax[ids], g.ay[ids]]),
                                    np.column_stack([g.bx[ids], g.by[ids]])], axis=1), **kw)


def file_sizes(folder, tid):
    return {ext: os.path.getsize(os.path.join(folder, f"{tid}.{ext}")) if os.path.exists(os.path.join(folder, f"{tid}.{ext}")) else None
            for ext in ("track", "nodes", "arcs", "route")}


def analyse_trip(partition, tid):
    folder = trip_dir(partition, tid)
    ti = load_trip_inputs(folder)
    route = load_reference_route(folder)
    tr, nodes, arcs = ti.track, ti.nodes, ti.arcs
    row = {"partition": partition, "trip": tid, **{f"bytes_{k}": v for k, v in file_sizes(folder, tid).items()}}
    row["n_obs"], row["n_nodes"], row["n_arcs"] = len(tr), len(nodes), len(arcs)
    row["nan_track"] = int(np.isnan(tr).sum())
    row["nan_nodes"] = int(np.isnan(nodes).sum())
    row["coord_out_of_range"] = int(((np.abs(tr[:, 0]) > 180) | (np.abs(tr[:, 1]) > 90)).sum())
    dt = np.diff(tr[:, 2])
    row["duration_s"] = float(tr[-1, 2] - tr[0, 2])
    row["dt_nonpositive"] = int((dt <= 0).sum())
    row["frac_dt_1s"] = float(np.mean(np.abs(dt - 1) < 1e-6)) if len(dt) else np.nan
    row["dt_max"] = float(dt.max()) if len(dt) else np.nan
    row["n_gaps_gt5s"] = int((dt > 5).sum())
    row["n_gaps_gt30s"] = int((dt > 30).sum())
    row["dup_positions"] = int((np.hypot(np.diff(tr[:, 0]), np.diff(tr[:, 1])) == 0).sum())
    step = haversine_m(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
    spd = step / np.maximum(dt, 1e-6)
    row["track_len_km"] = float(step.sum() / 1000)
    row["speed_p50_mps"] = float(np.median(spd))
    row["speed_max_mps"] = float(spd.max())
    row["n_speed_gt60mps"] = int((spd > 60).sum())
    row["lon_min"], row["lon_max"] = float(tr[:, 0].min()), float(tr[:, 0].max())
    row["lat_min"], row["lat_max"] = float(tr[:, 1].min()), float(tr[:, 1].max())
    row["extent_km_ew"] = float(haversine_m(row["lon_min"], row["lat_min"], row["lon_max"], row["lat_min"]) / 1000)
    row["extent_km_ns"] = float(haversine_m(row["lon_min"], row["lat_min"], row["lon_min"], row["lat_max"]) / 1000)

    # network checks
    row["arc_index_invalid"] = int(((arcs < 0) | (arcs >= len(nodes))).sum())
    row["self_loops"] = int((arcs[:, 0] == arcs[:, 1]).sum())
    pairs = set(map(tuple, arcs.tolist()))
    row["dup_arcs"] = len(arcs) - len(pairs)
    row["frac_arcs_with_reverse"] = float(np.mean([(b, a) in pairs for a, b in arcs.tolist()]))
    deg = np.bincount(arcs.ravel(), minlength=len(nodes))
    row["isolated_nodes"] = int((deg == 0).sum())
    A = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(len(nodes), len(nodes))).tocsr()
    nw, lw = connected_components(A, directed=True, connection="weak")
    ns, ls = connected_components(A, directed=True, connection="strong")
    row["weak_components"] = int(nw)
    row["largest_weak_frac"] = float(np.bincount(lw).max() / len(nodes))
    row["strong_components"] = int(ns)
    row["largest_strong_frac"] = float(np.bincount(ls).max() / len(nodes))

    proj = LocalProjection.for_track(tr[:, :2])
    pxy = proj.to_xy(tr[:, :2])
    g = RoadGraph(proj.to_xy(nodes), arcs)
    row["arc_len_median_m"] = float(np.median(g.length))
    row["zero_length_arcs"] = int((g.length < 1e-6).sum())
    row["has_route"] = route is not None
    if route is not None:
        row["route_len"] = len(route)
        row["route_unique"] = len(np.unique(route))
        row["route_invalid_ids"] = int(((route < 0) | (route >= len(arcs))).sum())
        brk = arcs[route[:-1], 1] != arcs[route[1:], 0]
        row["route_breaks"] = int(brk.sum())
        row["route_len_km"] = float(g.length[route].sum() / 1000)
        al = align_track_to_route(pxy, route, g)
        row["align_dist_p50"] = float(np.median(al["dist"]))
        row["align_dist_p95"] = float(np.percentile(al["dist"], 95))
        row["uncertain_labels"] = int((al["label"] < 0).sum())
        row["route_arcs_unvisited_frac"] = float(1 - len(np.unique(al["route_idx"])) / len(route))
        # does the route start / end near the track ends?
        row["start_gap_m"] = float(np.hypot(pxy[0, 0] - g.ax[route[0]], pxy[0, 1] - g.ay[route[0]]))
        row["end_gap_m"] = float(np.hypot(pxy[-1, 0] - g.bx[route[-1]], pxy[-1, 1] - g.by[route[-1]]))
        extra = {"align_dist": al["dist"], "route": route}
    else:
        extra = {}
    return row, {"tr": tr, "pxy": pxy, "g": g, "dt": dt, "spd": spd, "proj": proj, "ti": ti, **extra}


def main():
    rows, keep = [], {}
    all_dt, all_ad = {"train": [], "student_test": []}, {"train": [], "student_test": []}
    for part in ("train", "student_test"):
        for tid in list_trips(part):
            r, ex = analyse_trip(part, tid)
            rows.append(r)
            all_dt[part].append(ex["dt"])
            if "align_dist" in ex:
                all_ad[part].append(ex["align_dist"])
            keep[(part, tid)] = ex
            print(f"{part}/{tid}: n={r['n_obs']} arcs={r['n_arcs']} route={r.get('route_len')} "
                  f"breaks={r.get('route_breaks')} align_p95={r.get('align_dist_p95', float('nan')):.1f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "inventory.csv"), index=False)

    summary = {}
    for part, d in df.groupby("partition"):
        dts = np.concatenate(all_dt[part])
        ad = np.concatenate(all_ad[part])
        summary[part] = {
            "trips": int(len(d)), "observations": int(d.n_obs.sum()),
            "obs_per_trip": d.n_obs.describe().round(1).to_dict(),
            "missing_values_track": int(d.nan_track.sum()), "missing_values_nodes": int(d.nan_nodes.sum()),
            "coords_out_of_range": int(d.coord_out_of_range.sum()),
            "nonpositive_dt": int(d.dt_nonpositive.sum()),
            "frac_dt_exactly_1s": float(np.mean(np.abs(dts - 1) < 1e-6)),
            "dt_quantiles": {q: float(np.quantile(dts, q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)},
            "gaps_gt5s": int(d.n_gaps_gt5s.sum()), "gaps_gt30s": int(d.n_gaps_gt30s.sum()),
            "trips_with_route_breaks": int((d.route_breaks > 0).sum()),
            "total_route_breaks": int(d.route_breaks.sum()),
            "invalid_arc_or_route_ids": int(d.arc_index_invalid.sum() + d.route_invalid_ids.sum()),
            "isolated_nodes": int(d.isolated_nodes.sum()),
            "median_weak_components": float(d.weak_components.median()),
            "median_frac_arcs_with_reverse": float(d.frac_arcs_with_reverse.median()),
            "align_dist_quantiles": {q: float(np.quantile(ad, q)) for q in (0.5, 0.9, 0.95, 0.99)},
            "uncertain_labels": int(d.uncertain_labels.sum()),
            "countries_utm_zones": int(len({int((lo + 180) // 6) for lo in (d.lon_min + d.lon_max) / 2})),
        }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # ---------------- figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for part, c in (("train", "tab:blue"), ("student_test", "tab:orange")):
        d = df[df.partition == part]
        ax[0].hist(d.n_obs, bins=25, alpha=.6, label=part, color=c)
        ax[1].hist(d.route_len, bins=25, alpha=.6, label=part, color=c)
        ax[2].hist(d.track_len_km, bins=25, alpha=.6, label=part, color=c)
    for a, t in zip(ax, ("GPS observations per trip", "Reference route length (arcs)", "Track length (km)")):
        a.set_title(t); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "trip_size_distributions.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    dts = np.concatenate(all_dt["train"] + all_dt["student_test"])
    ax[0].hist(np.clip(dts, 0, 20), bins=np.arange(0, 21, 0.5), log=True)
    ax[0].set_title("Sampling interval (s), clipped at 20, log count  [train+student_test]")
    ad = np.concatenate(all_ad["train"])
    ax[1].hist(np.clip(ad, 0, 80), bins=80, log=True)
    ax[1].set_title("GPS-to-aligned-reference-arc distance (m) [train]")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sampling_and_noise.png"), dpi=130); plt.close(fig)

    # missing-value / anomaly bar chart
    cols = ["nan_track", "nan_nodes", "coord_out_of_range", "dt_nonpositive", "n_gaps_gt5s", "n_gaps_gt30s",
            "n_speed_gt60mps", "dup_positions", "isolated_nodes", "route_breaks", "uncertain_labels"]
    tot = df.groupby("partition")[cols].sum().T
    ax = tot.plot.barh(figsize=(9, 5), log=True)
    ax.set_title("Missing values and anomalies (counts, log scale)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "missing_and_anomalies.png"), dpi=130); plt.close()

    # trajectory grid (train), local metres
    tids = list_trips("train")[:12]
    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    for a, tid in zip(axs.ravel(), tids):
        ex = keep[("train", tid)]
        g, pxy = ex["g"], ex["pxy"]
        x0, x1 = pxy[:, 0].min() - 300, pxy[:, 0].max() + 300
        y0, y1 = pxy[:, 1].min() - 300, pxy[:, 1].max() + 300
        sel = ((g.ax > x0) & (g.ax < x1) & (g.ay > y0) & (g.ay < y1))
        a.add_collection(segs(g, sel, color="0.8", lw=.4))
        a.add_collection(segs(g, ex["route"], color="green", lw=2))
        a.plot(pxy[:, 0], pxy[:, 1], ".", ms=1.5, color="blue")
        a.set_title(f"train/{tid}"); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("GPS (blue), reference route (green), road network (grey) - UTM metres")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "trajectories_grid.png"), dpi=110); plt.close(fig)

    # whole network of one trip + degree distribution
    ex = keep[("train", tids[0])]
    g = ex["g"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].add_collection(segs(g, np.arange(g.n_arcs), color="0.5", lw=.2))
    ax[0].plot(ex["pxy"][:, 0], ex["pxy"][:, 1], color="blue", lw=1)
    ax[0].set_aspect("equal"); ax[0].set_title(f"Full road network supplied for train/{tids[0]}")
    ax[1].bar(df.trip[df.partition == "train"], df.n_arcs[df.partition == "train"])
    ax[1].set_xticks([]); ax[1].set_title("Arcs per trip network (train)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "network_overview.png"), dpi=120); plt.close(fig)

    # Folium map for two trips
    for part, tid in (("train", tids[0]), ("student_test", list_trips("student_test")[0])):
        ex = keep[(part, tid)]
        m = trip_map(ex["ti"], ex["g"], ex["proj"], reference=ex["route"], title=f"EDA {part}/{tid}")
        m.save(os.path.join(MAPS, f"eda_{part}_{tid}.html"))


if __name__ == "__main__":
    main()
