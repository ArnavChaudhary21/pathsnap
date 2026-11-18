"""Real epoch-wise training histories (only models that have them).

* MLP: trained epoch by epoch with partial_fit on GroupKFold fold 0; train and
  validation log loss recorded after every epoch.
* LightGBM (untuned and tuned): eval_set on fold 0 -> binary log loss after
  every boosting iteration.
Linear/NB/KNN/RF have no iterative training history, so no curve is drawn.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from pathsnap.data import stack
from pathsnap.io import list_trips
from pathsnap.models import make_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "comparison")
os.makedirs(OUT, exist_ok=True)

X, y, grp, _ = stack("train", list_trips("train"), 3)
tr, va = next(GroupKFold(n_splits=5).split(X, y, grp))
hist = {}

# ---- MLP
mlp_cfg_path = os.path.join(ROOT, "results", "experiments", "mlp.json")
pipe = make_model("mlp")
scaler, mlp = pipe.steps[0][1], pipe.steps[1][1]
Xtr, Xva = scaler.fit_transform(X[tr]), scaler.transform(X[va])
mlp.set_params(early_stopping=False)
rng = np.random.default_rng(352)
tl, vl = [], []
for ep in range(40):
    idx = rng.permutation(len(tr))
    for s in range(0, len(idx), mlp.batch_size):
        b = idx[s:s + mlp.batch_size]
        mlp.partial_fit(Xtr[b], y[tr][b], classes=[0, 1])
    tl.append(log_loss(y[tr], np.clip(mlp.predict_proba(Xtr)[:, 1], 1e-7, 1 - 1e-7)))
    vl.append(log_loss(y[va], np.clip(mlp.predict_proba(Xva)[:, 1], 1e-7, 1 - 1e-7)))
    print(f"mlp epoch {ep + 1}: train {tl[-1]:.4f} val {vl[-1]:.4f}", flush=True)
hist["mlp"] = {"train_logloss": tl, "val_logloss": vl}

# ---- LightGBM
for tag in ("lgbm", "lgbm_tuned"):
    p = os.path.join(ROOT, "results", "experiments", f"{tag}.json")
    if not os.path.exists(p):
        continue
    params = json.load(open(p))["config"]
    m = make_model("lgbm", params)
    m.fit(X[tr], y[tr], eval_set=[(X[tr], y[tr]), (X[va], y[va])], eval_names=["train", "val"],
          eval_metric="binary_logloss")
    ev = m.evals_result_
    hist[tag] = {"train_logloss": ev["train"]["binary_logloss"], "val_logloss": ev["val"]["binary_logloss"]}
    print(tag, "final train", hist[tag]["train_logloss"][-1], "val", hist[tag]["val_logloss"][-1])

json.dump(hist, open(os.path.join(OUT, "training_histories.json"), "w"))
fig, axs = plt.subplots(1, len(hist), figsize=(6 * len(hist), 4))
for ax, (k, h) in zip(np.atleast_1d(axs), hist.items()):
    ax.plot(range(1, len(h["train_logloss"]) + 1), h["train_logloss"], label="train (fold-0 training trips)")
    ax.plot(range(1, len(h["val_logloss"]) + 1), h["val_logloss"], label="validation (fold-0 held-out trips)")
    ax.set_xlabel("epoch" if k == "mlp" else "boosting iteration"); ax.set_ylabel("binary log loss")
    ax.set_title(k); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "training_curves.png"), dpi=130)
