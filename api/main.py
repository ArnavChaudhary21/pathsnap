"""PathSnap FastAPI backend (Laptop 1).

Run:
  set PATHSNAP_EVAL_DIR=<folder holding the 10 evaluation trip folders>
  uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
  GET  /health            liveness + loaded model info
  GET  /trips             evaluation trip ids (no labels)
  POST /predict           {"trip_id": "..."} -> predicted route, metrics, Folium map

The reference route of a trip is read only inside pathsnap.pipeline.evaluate_trip,
after prediction, and only aggregate scores leave the server.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.io import data_root
from pathsnap.pipeline import Predictor, evaluate_trip
from pathsnap.viz import trip_map

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.environ.get("PATHSNAP_ARTIFACTS", os.path.join(ROOT, "artifacts", "final"))
# Evaluation trips: $PATHSNAP_EVAL_DIR, else ../evaluator_test/evaluator_test (next to the
# repository), else the student_test partition.
_EVALUATOR_DEFAULT = os.path.join(os.path.dirname(ROOT), "evaluator_test", "evaluator_test")
EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR") or (
    _EVALUATOR_DEFAULT if os.path.isdir(_EVALUATOR_DEFAULT) else os.path.join(data_root(), "student_test"))
TRIP_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(os.path.join(ROOT, "results", "logs", "api_requests.log"))])
log = logging.getLogger("pathsnap.api")

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
predictor = Predictor(ARTIFACTS)
log.info("Loaded classifier=%s hmm=%s eval_dir=%s", predictor.manifest["classifier"],
         predictor.manifest["hmm_params"], EVAL_DIR)


def available_trips() -> list[str]:
    if not os.path.isdir(EVAL_DIR):
        return []
    return sorted(d for d in os.listdir(EVAL_DIR)
                  if TRIP_RE.match(d) and os.path.exists(os.path.join(EVAL_DIR, d, d + ".track")))


class PredictRequest(BaseModel):
    trip_id: str


def _dedup_arcs(pred):
    out = []
    for a in pred:
        if a >= 0 and (not out or out[-1] != a):
            out.append(int(a))
    return out


@app.get("/health")
def health():
    return {"status": "ok", "classifier": predictor.manifest["classifier"],
            "hmm_params": predictor.manifest["hmm_params"], "n_trips": len(available_trips())}


@app.get("/trips")
def trips():
    return {"eval_dir_name": os.path.basename(os.path.normpath(EVAL_DIR)), "trips": available_trips()}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    client = request.client.host if request.client else "?"
    if req.trip_id not in available_trips():
        log.warning("REJECT client=%s trip=%r (unknown trip)", client, req.trip_id)
        raise HTTPException(status_code=404, detail=f"Unknown trip id {req.trip_id!r}")
    folder = os.path.join(EVAL_DIR, req.trip_id)
    log.info("REQUEST client=%s trip=%s -> running candidates + classifier + Viterbi", client, req.trip_id)
    t0 = time.perf_counter()
    try:
        out = predictor.predict_trip(folder)          # inputs only
        metrics = evaluate_trip(folder, out)          # server-side reference, after prediction
    except Exception as e:  # noqa: BLE001
        log.exception("FAILED trip=%s", req.trip_id)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(e).__name__}: {e}")
    pt = out["pt"]
    arcs = _dedup_arcs(out["pred"])
    nodes, A = pt.ti.nodes, pt.ti.arcs
    route_geo = [[[float(nodes[A[a, 0], 1]), float(nodes[A[a, 0], 0])],
                  [float(nodes[A[a, 1], 1]), float(nodes[A[a, 1], 0])]] for a in arcs]
    # Folium map shows GPS + network + predicted route only (no reference route).
    m = trip_map(pt.ti, pt.g, pt.proj, predicted=out["pred"], title=f"Trip {req.trip_id} - predicted route")
    elapsed = time.perf_counter() - t0
    acc = metrics["hmm"]["top1_accuracy"] if metrics else None
    log.info("RESPONSE client=%s trip=%s n_obs=%d predicted_arcs=%d top1=%s time=%.2fs", client, req.trip_id,
             len(pt.pxy), len(arcs), f"{acc:.4f}" if acc is not None else "n/a", elapsed)
    return {
        "trip_id": req.trip_id,
        "n_observations": int(len(pt.pxy)),
        "n_predicted": int((out["pred"] >= 0).sum()),
        "predicted_arc_per_observation": [int(a) for a in out["pred"]],
        "predicted_route_arcs": arcs,
        "predicted_route_latlon": route_geo,
        "metrics": metrics,
        "timing_s": {**{k: round(v, 3) for k, v in out["timing"].items()}, "request_total_s": round(elapsed, 3)},
        "map_html": m.get_root().render(),
    }
