# Design decisions and supporting evidence

Each entry: decision, evidence (file that contains it), alternatives considered.

## D1. Per-trip UTM projection for all geometry
- **Evidence:** `results/eda/summary.json`. Train trips span 23 UTM zones, with longitudes from about 2° to about 136°. Trip extents are a few to a few tens of km.
- **Decision:** Project each trip to the UTM zone of its median GPS point (`pathsnap/geo.py`). Every distance, heading and projection is then computed in metres.
- **Why not degrees:** Euclidean distance in degrees mixes up east-west and north-south scale. For example, at 50° N one degree of longitude is about 71 km but one degree of latitude is 111 km.

## D2. Per-observation labels by monotone DP alignment
- **Evidence:** `.route` has far fewer rows than `.track` (for example 525 arcs vs 1,315 fixes in train/00000001), and it has no timestamps.
- **Decision:** Choose a non-decreasing route index j(i) for each GPS fix i. The objective is to minimise the sum of min(distance(fix, route[j]), 100 m). It is solved exactly by dynamic programming (`pathsnap/labels.py`). Route arcs that no fix lands on may be skipped, which happens for short arcs at speed.
- **Uncertain labels:** If the aligned arc is more than 50 m from the fix, the label is set to -1. Those points are excluded from training and from every metric denominator.
- **Evidence for the 50 m rule:** `results/eda/problem_zoom.png`. Trips 54 and 33 contain thousands of fixes in car parks, away from any road. Train has 23,163 uncertain points, about 81% of them from trip 54.
- **Limitation:** The labels are derived from the reference route. They are not independently verified ground truth.

## D3. Candidate set: radius 50 m, at most 10 nearest arcs, with fallback expansion
- **Evidence:** `results/candidate_recall_train.csv`.
  - Recall at 50 m with K=10 is 99.31%, with 6.5 candidates per fix on average.
  - K=15 adds only 0.18 pp of recall but costs about 20% more Viterbi work, which grows as O(K²).
  - Radii above 50 m give no extra recall, because labels beyond 50 m are uncertain by construction (D2).
- **Fallback:** If no arc is within 50 m, the search retries at 100 m and then 200 m. The `radius_level` feature records which radius was used.

## D4. Features (`pathsnap/features.py`)
These come from patterns seen in the EDA:
- **Parallel and two-way roads:** 95% of arcs have a reverse twin, so distance alone cannot tell the two directions apart. Features: `head_diff` and `cos_head` (GPS heading vs arc bearing), and `win_head_diff` (heading over a ±5 fix window).
- **GPS noise:** The median offset from the road is 5.7 m and the 95th percentile on non-parked fixes is under 20 m. Features: the ±5 fix context distance `win_dist` / `win_dist_far`, so a single noisy fix does not decide the result.
- **Relative features:** `dist_rel`, `head_diff_rel`, `win_dist_rel` and `dist_rank`. The classifier scores each candidate on its own, so these features carry "compared with the other candidates of the same fix".
- **Junction complexity:** `out_deg_to`, `in_deg_from`, `n_cand` and `log_n_cand_20m`.
- **Arc ends:** `off_end`, the overshoot beyond the arc end. When the projection is clamped, the fix is really near the next arc.
- **Gaps:** `dt_prev` and `dt_next`, since 1,216 train gaps are longer than 5 s.
- **Motion:** `speed` and `turn_rate`. Heading is unreliable when the vehicle is slow.
- **Label-free:** No feature uses the route or labels, so the same code runs at inference.

## D5. Training subsample: every 3rd observation
- **Reason:** At 1 Hz, consecutive fixes are almost duplicates. Every 3rd fix keeps all 80 trips and about 420k rows, and makes the 5-fold CV of six models affordable.
- **Evaluation:** Student-test evaluation uses all observations.

## D6. Validation protocol
- 5-fold GroupKFold with groups = trip, so no trip is split across folds.
- Scalers and transformers are inside each sklearn pipeline, so they are fitted on the training folds only.
- The 10 student-test trips are used only for development-stage comparison.
- The evaluator partition is never used.

## D7. Primary selection metric
- **Metric:** CV Macro F1, as the official guidelines require. Student-test Top-1 accuracy and log loss are secondary.
- **Decision rule for Macro F1:** The top-scoring candidate of each fix counts as the positive prediction. This is the decision the system actually makes, and it does not depend on a 0.5 probability threshold.
- **Why log loss matters:** It measures how good the probabilities are, which matters because they become HMM emissions.

## D8. HMM design (`pathsnap/hmm.py`)
- **Hidden states:** The candidate arcs, together with each fix's projected position on the arc.
- **Emissions:** The classifier probability, normalised within the candidate set.
- **Transitions:** An exponential model in |network distance − GPS distance|. This follows the formulation of Newson & Krumm (2009). Network distance comes from a bounded Dijkstra that respects arc direction.
- **Unreachable transitions:** These get a finite penalty instead of −∞. The reason is that 13 route breaks were found in train references, and the networks have many weak components (median 20). The decoder must survive network breaks rather than fail.
- **After GPS gaps:** beta grows with dt, so the model is more tolerant of route changes that happened during the gap.
- **Decoder tuning:** Parameters are tuned on train trips using out-of-fold classifier probabilities, so the emissions are not optimistic.

## D9. Deployment
- **Map rendering:** The API renders the Folium map server-side and returns it as HTML. The UI shows the HTML in an iframe, so the UI needs no mapping code of its own.
- **Ground truth:** The reference route is read only in `pipeline.evaluate_trip`, after prediction. Only aggregate scores are returned to the UI.

## D10. Bayesian optimisation outcome (LightGBM)
- **Search:** 40 Optuna TPE trials (18 complete, 22 pruned by MedianPruner), objective = mean 5-fold GroupKFold Macro F1 on the 80 train trips. Wall time of the final resumed run: about 20 min.
- **Result:** CV Macro F1 0.9471 (tuned) vs 0.9462 (untuned). On student test, Top-1 is tied (0.9060 vs 0.9064) and log loss is worse (0.0946 vs 0.0722).
- **Why:** The tuned configuration (num_leaves 197, learning_rate 0.118) fits the training trips almost perfectly. `results/comparison/training_curves.png` shows its fold-0 validation log loss rising after about 40 boosting iterations while training log loss goes to 0.0015. The objective optimised F1 (a ranking decision), not calibration, so this overfitting is not penalised by the objective.
- **Decision:** `lgbm_tuned` is deployed, because model selection follows the declared primary metric (CV Macro F1, D7). Its probabilities still decode well (D11).

## D11. Deployed HMM decoder parameters
- **Tuning:** 30 TPE trials on 25 train trips with out-of-fold probabilities raised HMM Top-1 there from 0.9379 (default parameters) to 0.9447 (tuned). Tuned beta ended at 59, the edge of its 1-60 range.
- **Student test:** Default parameters gave Top-1 0.9271, continuity 0.9987, route F1 0.9787. Tuned parameters gave 0.9222, 0.9952, 0.9750. The tuned decoder did not transfer.
- **Decision:** `scripts/06_hmm_eval.py` deploys whichever setting has the higher student-test HMM Top-1, which is the default. The guidelines allow decoder choices on training and student-test data; the evaluator partition is not used.

## D12. Evaluator partition
- `scripts/10_evaluator_final_eval.py` scores the frozen `artifacts/final` pipeline on the 10 evaluator trips once, for the report. No model, feature, hyperparameter, decoder or threshold choice was made after it ran.
- The API serves `../evaluator_test/evaluator_test` by default when that folder exists (override with `PATHSNAP_EVAL_DIR`).
