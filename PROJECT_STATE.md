# Project state / continuation handoff

Last updated: 2026-11-18. Status by checkpoint:

| Checkpoint | Status | Evidence |
|---|---|---|
| 1. EDA, labels | DONE | `results/eda/*`, `results/maps/eda_*.html`, `results/eda_log.txt` |
| 2. Candidates, features, 6 models | DONE | `results/experiments/*.json`, `results/logs/train_*.log`, `results/features/*.csv` (local only, >100 MB) |
| 3a. Bayesian optimisation (lgbm, 40 trials) | DONE | `results/bayes_opt/`, `results/experiments/lgbm_tuned.json`, `results/logs/bayes_opt_lgbm.log` |
| 3b. HMM/Viterbi tuning + evaluation | DONE | `results/hmm/`, `results/maps/student_test_*_hmm_*.html`, `results/logs/hmm_eval.log` |
| Comparison plots, training curves | DONE | `results/comparison/` |
| Final artifacts | DONE | `artifacts/final/classifier.joblib`, `artifacts/final/manifest.json` |
| Held-out evaluator score | DONE | `results/final_eval/` |
| 4. API + UI | DONE; API tested locally on all 10 evaluator trips | `api/main.py`, `ui/`, `results/logs/api_requests.log` |
| 5. Repo + report | TODO: git init + push; team writes the PDF report | |
| 6. Two-laptop test on real hardware | TODO | |

## Remaining steps (team)
1. Two-laptop rehearsal: Laptop 1 runs the API (`python -m uvicorn api.main:app --host 0.0.0.0 --port 8000`), Laptop 2 serves `ui/` and connects to `http://<Laptop-1-IP>:8000`. Take screenshots of the UI and of the API terminal log for report section 7.
2. Write the PDF report from the results listed in `README.md` and `DECISIONS.md`.
3. `git init`, commit, push to a public GitHub repository.
