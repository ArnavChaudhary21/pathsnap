"""Folium maps. Note: our arrays are (lon, lat); Folium wants (lat, lon)."""
from __future__ import annotations

import folium
import numpy as np
from branca.element import Element


def _dedup(seq):
    out = []
    for a in seq:
        if a >= 0 and (not out or out[-1] != a):
            out.append(int(a))
    return out


def _arc_lines(arcs_ids, ti):
    n, a = ti.nodes, ti.arcs
    return [[[n[a[k, 0], 1], n[a[k, 0], 0]], [n[a[k, 1], 1], n[a[k, 1], 0]]] for k in arcs_ids]


def trip_map(ti, g, proj, reference=None, predicted=None, classifier_only=None,
             title="PathSnap", network_margin_m=150.0, max_network_arcs=15000):
    tr = ti.track
    lat, lon = tr[:, 1], tr[:, 0]
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])

    # road network near the track
    pxy = proj.to_xy(tr[:, :2])
    x0, y0 = pxy.min(0) - network_margin_m
    x1, y1 = pxy.max(0) + network_margin_m
    mx, my = (g.ax + g.bx) / 2, (g.ay + g.by) / 2
    sel = np.flatnonzero((mx > x0) & (mx < x1) & (my > y0) & (my < y1))
    if len(sel) > max_network_arcs:
        d = np.min(np.hypot(mx[sel, None] - pxy[None, ::20, 0], my[sel, None] - pxy[None, ::20, 1]), axis=1)
        sel = sel[np.argsort(d)[:max_network_arcs]]
    fg = folium.FeatureGroup(name="Road network (nearby arcs)")
    folium.PolyLine(_arc_lines(sel, ti), color="#888888", weight=1, opacity=0.6).add_to(fg)
    fg.add_to(m)

    if reference is not None:
        fg = folium.FeatureGroup(name="Reference route (server-side)")
        folium.PolyLine(_arc_lines(_dedup(reference), ti), color="#1a9641", weight=6, opacity=0.7).add_to(fg)
        fg.add_to(m)
    if classifier_only is not None:
        fg = folium.FeatureGroup(name="Classifier-only arcs", show=False)
        folium.PolyLine(_arc_lines(_dedup(classifier_only), ti), color="#fdae61", weight=3, opacity=0.9).add_to(fg)
        fg.add_to(m)
    if predicted is not None:
        fg = folium.FeatureGroup(name="Predicted route (HMM/Viterbi)")
        folium.PolyLine(_arc_lines(_dedup(predicted), ti), color="#d7191c", weight=3, opacity=0.9).add_to(fg)
        fg.add_to(m)

    fg = folium.FeatureGroup(name="Raw GPS observations")
    step = max(1, len(tr) // 3000)
    for k in range(0, len(tr), step):
        folium.CircleMarker([lat[k], lon[k]], radius=2, color="#2b83ba", fill=True, weight=0,
                            fill_opacity=0.8).add_to(fg)
    fg.add_to(m)
    folium.Marker([lat[0], lon[0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    rows = [("#2b83ba", "Raw GPS observations"), ("#888888", "Road network")]
    if reference is not None:
        rows.append(("#1a9641", "Reference route"))
    if classifier_only is not None:
        rows.append(("#fdae61", "Classifier-only"))
    if predicted is not None:
        rows.append(("#d7191c", "Predicted route"))
    items = "".join(f'<div><span style="display:inline-block;width:18px;height:4px;background:{c};'
                    f'margin-right:6px;vertical-align:middle"></span>{t}</div>' for c, t in rows)
    legend = (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;'
              f'border:1px solid #999;border-radius:4px;font:12px sans-serif"><b>{title}</b>{items}</div>')
    m.get_root().html.add_child(Element(legend))
    return m
