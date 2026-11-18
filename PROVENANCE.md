# Provenance and attribution

## Third-party libraries
These are used as dependencies, not copied into this repository. Licences are as published by each project.

| Library | Use | Licence |
|---|---|---|
| numpy, scipy | arrays, KD-tree (`cKDTree`), connected components | BSD-3 |
| scikit-learn | LogisticRegression, GaussianNB, KNN, RandomForest, MLP, GroupKFold, metrics | BSD-3 |
| LightGBM | gradient boosting classifier | MIT |
| Optuna | TPE Bayesian optimisation, SQLite trial storage | MIT |
| pyproj | WGS84 to UTM projection | MIT |
| folium / branca | interactive Leaflet maps | MIT |
| FastAPI, uvicorn, pydantic | REST API | MIT / BSD |
| matplotlib, pandas, joblib | plots, tables, model persistence | PSF-based / BSD |

## Algorithms and literature
- **HMM map matching formulation:** P. Newson and J. Krumm, "Hidden Markov Map Matching Through Noise and Sparseness", ACM SIGSPATIAL GIS 2009. We follow its idea of an exponential transition probability based on the difference between route distance and great-circle distance.
  - Our implementation in `pathsnap/hmm.py` was written for this project.
  - Our emissions come from a trained classifier, not from a Gaussian distance model.
  - We add a finite break penalty for unreachable transitions and gap-dependent beta.
- **Viterbi algorithm:** A. Viterbi (1967). A standard dynamic-programming recurrence, implemented in `viterbi_core` and tested against brute-force enumeration.
- **TPE sampler:** J. Bergstra et al., "Algorithms for Hyper-Parameter Optimization", NeurIPS 2011. Used through Optuna.
- **Monotone label alignment:** `pathsnap/labels.py` is a constrained dynamic-programming alignment designed for this dataset. It is related in spirit to dynamic time warping.

## Dataset
- **Source:** Supplied by the course organisers (UE24CS352A PathSnap hackathon).
- **Not redistributed:** The data is not included in this repository (see `.gitignore`). Point `PATHSNAP_DATA` at the supplied `Student folder`.

## AI assistance disclosure
- **Tool:** The code, experiment scripts and documentation drafts were produced with help from an AI coding assistant (Claude, by Anthropic), working in the team's environment on the team's machine.
- **Real runs:** Every metric, table and plot in this repository was produced by actually executing the committed scripts on the supplied data. The logs are in `results/logs/` and the machine-readable results are in `results/`.
- **Responsibility:** Team members are responsible for reviewing, understanding and being able to explain all of this material, as the course's AI-use policy requires. Declare this assistance in the report as the event rules require.
