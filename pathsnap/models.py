"""Model zoo: one candidate-scoring classifier per algorithm family.

All models share the same input (candidate feature rows, features.FEATURES)
and the same target (row is the reference arc). Scaling is part of each
pipeline so it is fitted only on the training fold.
"""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

SEED = 352

# Initial ("untuned") configurations. Recorded verbatim in results/experiments.
CONFIGS = {
    "logreg": {"family": "Linear / probabilistic", "C": 1.0, "max_iter": 2000},
    "gnb": {"family": "Probabilistic (generative)", "var_smoothing": 1e-9, "quantile_normal": True},
    "knn": {"family": "Instance-based", "n_neighbors": 50, "weights": "distance"},
    "rf": {"family": "Tree ensemble (bagging)", "n_estimators": 200, "min_samples_leaf": 5,
           "max_features": "sqrt"},
    "lgbm": {"family": "Boosting", "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63,
             "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 0.0},
    "mlp": {"family": "Neural network", "hidden_layer_sizes": (64, 32), "alpha": 1e-4,
            "learning_rate_init": 1e-3, "batch_size": 512, "max_iter": 60, "early_stopping": True},
}

LOSS = {
    "logreg": "L2-regularised binary cross-entropy (lbfgs)",
    "gnb": "Maximum likelihood of class-conditional Gaussians (no iterative loss)",
    "knn": "None (lazy learner); probability = distance-weighted neighbour vote",
    "rf": "Gini impurity per split; probabilities = mean leaf class frequency",
    "lgbm": "Binary log-loss (gradient boosting on second-order Taylor expansion)",
    "mlp": "Binary cross-entropy + L2 penalty (Adam)",
}


def make_model(name: str, params: dict | None = None):
    c = {**CONFIGS[name], **(params or {})}
    c = {k: v for k, v in c.items() if k != "family"}
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=c["C"], max_iter=c["max_iter"]))
    if name == "gnb":
        steps = [QuantileTransformer(output_distribution="normal", subsample=200_000, random_state=SEED)] \
            if c["quantile_normal"] else [StandardScaler()]
        return make_pipeline(*steps, GaussianNB(var_smoothing=c["var_smoothing"]))
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=c["n_neighbors"], weights=c["weights"], algorithm="brute", n_jobs=-1))
    if name == "rf":
        return RandomForestClassifier(n_estimators=c["n_estimators"], min_samples_leaf=c["min_samples_leaf"],
                                      max_features=c["max_features"], n_jobs=-1, random_state=SEED)
    if name == "lgbm":
        return LGBMClassifier(n_estimators=c["n_estimators"], learning_rate=c["learning_rate"],
                              num_leaves=c["num_leaves"], min_child_samples=c["min_child_samples"],
                              subsample=c["subsample"], subsample_freq=1 if c["subsample"] < 1 else 0,
                              colsample_bytree=c["colsample_bytree"], reg_lambda=c["reg_lambda"],
                              random_state=SEED, n_jobs=-1, verbose=-1)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=tuple(c["hidden_layer_sizes"]), alpha=c["alpha"],
            learning_rate_init=c["learning_rate_init"], batch_size=c["batch_size"], max_iter=c["max_iter"],
            early_stopping=c["early_stopping"], validation_fraction=0.1, n_iter_no_change=8,
            random_state=SEED))
    raise KeyError(name)
