# PathSnap: Map-Matching and Route Reconstruction

**Course:** UE24CS352A Machine Learning hackathon, PES University.

**Pipeline:** noisy GPS fixes → candidate road arcs → ML candidate classifier → HMM/Viterbi route reconstruction → FastAPI backend (Laptop 1) → evaluator UI (Laptop 2).

## Repository layout

| Path | Contents |
|---|---|
| `pathsnap/io.py` | raw readers. `load_trip_inputs` (track/nodes/arcs) is kept separate from `load_reference_route` |
| `pathsnap/geo.py` | per-trip UTM projection, point-to-segment geometry |
| `pathsnap/graph.py` | road graph: spatial index, bounded Dijkstra network distance |
| `pathsnap/labels.py` | per-observation labels from the chronological route (monotone DP alignment) |
| `pathsnap/features.py` | candidate generation and 22 candidate features |
| `pathsnap/models.py` | the 6 classifiers (6 algorithm families) and their initial configurations |
| `pathsnap/metrics.py` | Top-1, Macro F1, log loss, candidate miss rate, route continuity, route F1 |
| `pathsnap/hmm.py` | HMM emissions/transitions and Viterbi decoding |
| `pathsnap/pipeline.py` | inference (`Predictor`) and server-side evaluation (`evaluate_trip`) |
| `pathsnap/viz.py` | Folium maps |
| `scripts/01_eda.py` | dataset exploration → `results/eda/`, `results/maps/eda_*.html` |
| `scripts/02_candidate_recall.py` | candidate radius/K study → `results/candidate_recall_train.csv` |
| `scripts/03_build_dataset.py` | candidate + feature cache (`cache/`, git-ignored, rebuilt in about 20 s) |
| `scripts/04_train_models.py` | group-aware CV and student-test evaluation per model → `results/experiments/` |
| `scripts/05_bayes_opt.py` | Optuna TPE Bayesian optimisation → `results/bayes_opt/` |
| `scripts/06_hmm_eval.py` | decoder tuning, classifier vs HMM comparison, maps, export of `artifacts/final/` |
| `scripts/07_compare_models.py` | comparison table and plots → `results/comparison/` |
| `scripts/08_training_curves.py` | real epoch-wise loss curves (MLP, LightGBM) → `results/comparison/training_curves.png` |
| `scripts/09_export_features_csv.py` | exports the engineered feature dataset to CSV → `results/features/{train,student_test}_features.csv` |
| `scripts/10_evaluator_final_eval.py` | one-off held-out score of the frozen pipeline on the evaluator trips → `results/final_eval/` |
| `api/main.py` | FastAPI backend |
| `ui/` | evaluator UI (static HTML/JS) |
| `tests/` | unit tests: Viterbi vs brute force, network distance, HMM correction, alignment |
| `artifacts/final/` | deployed classifier + `manifest.json` (features, candidate config, HMM params) |
| `DECISIONS.md`, `PROVENANCE.md`, `SECURITY_AUDIT.md` | design rationale, attribution and AI disclosure, security review |

## Installation (both laptops)

You need Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The dataset is not included in this repository. Point the code at the supplied folder; it must contain `train/` and `student_test/`:

```bash
set PATHSNAP_DATA=C:\path\to\Student folder\Student folder     # PowerShell: $env:PATHSNAP_DATA="..."
```

If you don't set `PATHSNAP_DATA`, the code looks for `../Student folder/Student folder` relative to the repository.

## Reproducing the experiments

Run these in order:

```bash
python scripts/01_eda.py
python scripts/02_candidate_recall.py
python scripts/03_build_dataset.py
python scripts/04_train_models.py logreg gnb knn rf lgbm mlp
python scripts/05_bayes_opt.py lgbm 40
python scripts/06_hmm_eval.py lgbm_tuned 30
python scripts/07_compare_models.py
python scripts/08_training_curves.py
python scripts/09_export_features_csv.py      # optional: feature CSVs
python scripts/10_evaluator_final_eval.py     # once, after artifacts/final is frozen
python -m pytest -q tests
```

All randomness is seeded (`SEED = 352` in `pathsnap/models.py`). The Optuna studies are stored in SQLite, so a rerun resumes the same study.

## Results summary

Final system: LightGBM (Bayes-tuned) candidate classifier + HMM/Viterbi decoder (default parameters, see `DECISIONS.md` D10-D11).

| Partition | Classifier-only Top-1 | HMM Top-1 | HMM route continuity | HMM route F1 |
|---|---|---|---|---|
| student_test (10 trips) | 0.9060 | 0.9271 | 0.9987 | 0.9787 |
| evaluator_test (10 trips, held out) | 0.9094 | 0.9490 | 0.9988 | 0.9653 |

Sources: `results/hmm/hmm_results.json`, `results/final_eval/evaluator_test_summary.json`. Per-model comparison: `results/comparison/model_comparison.md`.

## Two-laptop deployment

### Laptop 1: model + API
1. Find the LAN IP. On Windows run `ipconfig` and look for the IPv4 address, for example `192.168.1.20`.
2. Put the ten evaluation trip folders in one directory, each laid out as `<id>/<id>.track/.nodes/.arcs/.route`. Point the API at it:

   ```powershell
   $env:PATHSNAP_EVAL_DIR = "C:\path\to\evaluator_test"
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

   If `PATHSNAP_EVAL_DIR` is not set, the API serves `../evaluator_test/evaluator_test` when it exists, otherwise the 10 `student_test` trips.
3. Allow inbound port 8000 through the firewall. Run this once, in an administrator PowerShell:

   ```powershell
   New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
   ```

4. Keep this terminal visible. Every request is logged with the client IP, trip, accuracy and time. The same lines are also written to `results/logs/api_requests.log`.

### Laptop 2: UI
1. Serve the UI folder:

   ```bash
   cd ui
   python -m http.server 5500
   ```

2. Open `http://localhost:5500` in a browser.
3. Enter `http://<Laptop-1-IP>:8000` in the API URL box and click **Connect**. You can instead set it in `ui/config.js`.
4. The ten evaluation trips are listed. Click one: its trip ID is sent to `POST /predict`, and the predicted route map and the scores are displayed.

**Quick connectivity check from Laptop 2:**

```bash
curl http://<Laptop-1-IP>:8000/health
```

### API

| Endpoint | Purpose |
|---|---|
| `GET /health` | model and HMM parameters loaded |
| `GET /trips` | ids of the evaluation trips available on the server |
| `POST /predict` `{"trip_id": "..."}` | full pipeline. Returns the predicted arc for every observation, the deduplicated predicted route (lat/lon), the metrics, timings, and the Folium map HTML |

**Ground-truth handling:** The reference route is read only by `pipeline.evaluate_trip`, after the prediction is complete. Only aggregate scores are returned, and the map sent to the UI does not contain the reference route.
