"""Checkpoint 3a: Bayesian hyperparameter optimisation of the selected model.

  python scripts/05_bayes_opt.py lgbm 40

* Sampler: Optuna TPE (Tree-structured Parzen Estimator) - a sequential
  model-based (Bayesian) optimiser; seeded for reproducibility.
* Objective: mean 5-fold GroupKFold (groups = trip) Macro F1 on the 80
  training trips - identical folds/data to scripts/04_train_models.py.
* Every trial is persisted in results/bayes_opt/<model>_study.db (SQLite)
  and exported to <model>_trials.csv.
* The best configuration is then refit and evaluated exactly like the
  untuned model (tag <model>_tuned).
Student-test and evaluator data are never used inside the objective.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import optuna
from sklearn.model_selection import GroupKFold

import importlib.util

from pathsnap.data import stack
from pathsnap.io import list_trips
from pathsnap.models import SEED, make_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "bayes_opt")
os.makedirs(OUT, exist_ok=True)
_spec = importlib.util.spec_from_file_location("train04", os.path.join(ROOT, "scripts", "04_train_models.py"))
train04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train04)

SPACES = {
    "lgbm": lambda tr: {
        "n_estimators": tr.suggest_int("n_estimators", 100, 1200, log=True),
        "learning_rate": tr.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": tr.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": tr.suggest_int("min_child_samples", 5, 300, log=True),
        "subsample": tr.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": tr.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": tr.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    },
    "rf": lambda tr: {
        "n_estimators": tr.suggest_int("n_estimators", 100, 500, log=True),
        "min_samples_leaf": tr.suggest_int("min_samples_leaf", 1, 50, log=True),
        "max_features": tr.suggest_float("max_features", 0.2, 0.9),
    },
    "mlp": lambda tr: {
        "hidden_layer_sizes": [tr.suggest_int("h1", 16, 256, log=True), tr.suggest_int("h2", 8, 128, log=True)],
        "alpha": tr.suggest_float("alpha", 1e-6, 1e-2, log=True),
        "learning_rate_init": tr.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
    },
}


def main(name, n_trials):
    X, y, grp, obs = stack("train", list_trips("train"), train04.STRIDE)
    folds = list(GroupKFold(n_splits=train04.N_FOLDS).split(X, y, grp))

    def objective(trial):
        params = SPACES[name](trial)
        scores = []
        for k, (tr, va) in enumerate(folds):
            m = make_model(name, params)
            m.fit(X[tr], y[tr])
            s = train04.fold_scores(m, X[va], y[va], obs[va])
            scores.append(s["macro_f1"])
            trial.set_user_attr(f"fold{k}_top1", s["top1_acc"])
            trial.set_user_attr(f"fold{k}_logloss", s["log_loss"])
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_std", float(np.std(scores)))
        return float(np.mean(scores))

    study = optuna.create_study(
        study_name=f"{name}_cv_macro_f1", direction="maximize",
        storage=f"sqlite:///{os.path.join(OUT, name + '_study.db')}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
    t0 = time.time()
    remaining = n_trials - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False,
                       callbacks=[lambda s, t: print(f"trial {t.number} {t.state.name} value={t.value} "
                                                     f"params={t.params}", flush=True)])
    wall = time.time() - t0
    study.trials_dataframe().to_csv(os.path.join(OUT, f"{name}_trials.csv"), index=False)
    best = study.best_trial
    params = SPACES[name](optuna.trial.FixedTrial(best.params))
    summary = {"model": name, "sampler": "TPE (multivariate, seed=%d)" % SEED, "pruner": "MedianPruner",
               "objective": "mean 5-fold GroupKFold Macro F1 (train trips, stride 3)",
               "n_trials_total": len(study.trials),
               "n_complete": len([t for t in study.trials if t.state.name == "COMPLETE"]),
               "n_pruned": len([t for t in study.trials if t.state.name == "PRUNED"]),
               "wall_seconds_this_run": wall, "best_value": best.value, "best_trial": best.number,
               "best_params": params}
    with open(os.path.join(OUT, f"{name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    train04.run(name, params, tag=f"{name}_tuned")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
