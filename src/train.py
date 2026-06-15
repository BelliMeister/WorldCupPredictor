"""
Train models with Optuna hyperparameter tuning.
Run: python src/train.py [--quick]
  --quick: skip Optuna, use defaults (faster, ~2 min)
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, StackingClassifier
import xgboost as xgb
import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

from features import FEATURE_COLS, engineer_features, match_weight

DATA_PATH = "data/raw/international_results.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

CUTOFF_YEAR = 1994
OPTUNA_TRIALS = 40


# ── Preprocessing step shared by all models ────────────────────────────────────

def make_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


# ── Optuna objective ───────────────────────────────────────────────────────────

def _xgb_objective(trial, X, y, weights, tscv):
    # n_estimators fixed at 500; early stopping finds the right number automatically
    params = {
        "n_estimators": 500,
        "early_stopping_rounds": 30,
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "gamma": trial.suggest_float("gamma", 0, 1.5),
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    preprocessor = make_preprocessor()
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = weights[train_idx]

        X_tr_p = preprocessor.fit_transform(X_tr)
        X_val_p = preprocessor.transform(X_val)

        clf = xgb.XGBClassifier(**params)
        clf.fit(X_tr_p, y_tr, sample_weight=w_tr,
                eval_set=[(X_val_p, y_val)], verbose=False)
        probs = clf.predict_proba(X_val_p)
        scores.append(-log_loss(y_val, probs))

    return np.mean(scores)


def _lgb_objective(trial, X, y, weights, tscv):
    # n_estimators fixed at 500; early stopping finds the right number automatically
    params = {
        "n_estimators": 500,
        "num_leaves": trial.suggest_int("num_leaves", 20, 80),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }

    preprocessor = make_preprocessor()
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = weights[train_idx]

        X_tr_p = preprocessor.fit_transform(X_tr)
        X_val_p = preprocessor.transform(X_val)

        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_tr_p, y_tr, sample_weight=w_tr,
                eval_set=[(X_val_p, y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        probs = clf.predict_proba(X_val_p)
        scores.append(-log_loss(y_val, probs))

    return np.mean(scores)


def tune_model(name: str, objective, X, y, weights, tscv, n_trials: int) -> dict:
    print(f"  Tuning {name} ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, X, y, weights, tscv), n_trials=n_trials, show_progress_bar=False)
    print(f"    Best log-loss: {-study.best_value:.4f}")
    return study.best_params


# ── Model builders ─────────────────────────────────────────────────────────────

def build_stacking_model(xgb_params: dict, lgb_params: dict) -> Pipeline:
    xgb_clf = xgb.XGBClassifier(
        **{k: v for k, v in xgb_params.items() if k not in ("objective", "num_class", "eval_metric", "random_state", "n_jobs")},
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    lgb_clf = lgb.LGBMClassifier(
        **{k: v for k, v in lgb_params.items() if k not in ("objective", "num_class", "metric", "random_state", "verbose", "n_jobs")},
        objective="multiclass",
        num_class=3,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    lr_meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    stacker = StackingClassifier(
        estimators=[("xgb", xgb_clf), ("lgb", lgb_clf)],
        final_estimator=lr_meta,
        cv=3,
        passthrough=True,
        n_jobs=-1,
    )

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", stacker),
    ])


def build_default_model() -> Pipeline:
    """Fast default model without Optuna tuning."""
    xgb_clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        random_state=42, n_jobs=-1,
    )
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=400, num_leaves=50, learning_rate=0.05,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        objective="multiclass", num_class=3,
        random_state=42, verbose=-1, n_jobs=-1,
    )
    lr_meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    stacker = StackingClassifier(
        estimators=[("xgb", xgb_clf), ("lgb", lgb_clf)],
        final_estimator=lr_meta,
        cv=3,
        passthrough=True,
        n_jobs=-1,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", stacker),
    ])


def build_goals_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)),
    ])


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "log_loss": log_loss(y_test, probs),
        "auc_ovr": roc_auc_score(y_test, probs, multi_class="ovr"),
    }


def evaluate_confident(model, X_test, y_test, threshold: float = 0.55) -> dict:
    """Accuracy on predictions where model confidence ≥ threshold."""
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    mask = confidence >= threshold
    if mask.sum() == 0:
        return {"accuracy_confident": np.nan, "coverage": 0.0}
    return {
        "accuracy_confident": accuracy_score(y_test[mask], preds[mask]),
        "coverage": mask.mean(),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip Optuna tuning")
    args = parser.parse_args()

    print("Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[df["date"].dt.year >= CUTOFF_YEAR].reset_index(drop=True)
    df = df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    print(f"  {len(df):,} matches ({CUTOFF_YEAR}–present)")

    print("Engineering features...")
    df = engineer_features(df, attach_rankings=True)
    print(f"  Features ready. Shape: {df.shape}")

    X = df[FEATURE_COLS].values
    y = df["outcome"].values
    weights = df["match_weight"].values
    y_goals = df["total_goals"].values

    # Time-series CV — last fold = most recent data for eval
    tscv = TimeSeriesSplit(n_splits=5)
    folds = list(tscv.split(X))
    train_idx, test_idx = folds[-1]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    w_train = weights[train_idx]
    y_train_goals, y_test_goals = y_goals[train_idx], y_goals[test_idx]

    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    if args.quick:
        print("\nBuilding default stacking model (--quick mode)...")
        outcome_model = build_default_model()
        outcome_model.fit(X_train, y_train)
    else:
        # Tune both base models with Optuna on inner CV
        inner_tscv = TimeSeriesSplit(n_splits=2)
        xgb_params = tune_model("XGBoost", _xgb_objective, X_train, y_train, w_train, inner_tscv, OPTUNA_TRIALS)
        lgb_params = tune_model("LightGBM", _lgb_objective, X_train, y_train, w_train, inner_tscv, OPTUNA_TRIALS)

        print("\nBuilding tuned stacking model...")
        outcome_model = build_stacking_model(xgb_params, lgb_params)
        outcome_model.fit(X_train, y_train)

    print("Training goals model...")
    goals_model = build_goals_model()
    goals_model.fit(X_train, y_train_goals, reg__sample_weight=w_train)

    metrics = evaluate(outcome_model, X_test, y_test)
    conf_metrics = evaluate_confident(outcome_model, X_test, y_test, threshold=0.50)
    goals_mae = np.mean(np.abs(goals_model.predict(X_test) - y_test_goals))

    print("\n=== Outcome Model (held-out test) ===")
    print(f"  Accuracy (all)      : {metrics['accuracy']:.3f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"  Accuracy (conf≥50%) : {conf_metrics['accuracy_confident']:.3f}  "
          f"({conf_metrics['accuracy_confident']*100:.1f}%, coverage {conf_metrics['coverage']:.0%})")
    print(f"  Log Loss            : {metrics['log_loss']:.4f}")
    print(f"  AUC (OvR)           : {metrics['auc_ovr']:.4f}")
    print(f"\n  Goals MAE           : {goals_mae:.3f}")

    # Retrain on ALL data before saving
    print("\nRetraining on full dataset...")
    outcome_model.fit(X, y)
    goals_model.fit(X, y_goals)

    with open(MODEL_DIR / "outcome_model.pkl", "wb") as f:
        pickle.dump(outcome_model, f)
    with open(MODEL_DIR / "goals_model.pkl", "wb") as f:
        pickle.dump(goals_model, f)

    # Save Elo snapshot for inference
    elo_snapshot: dict[str, float] = {}
    for _, row in df.iterrows():
        elo_snapshot[row["home_team"]] = row["home_elo_before"]
        elo_snapshot[row["away_team"]] = row["away_elo_before"]

    with open(MODEL_DIR / "elo_ratings.pkl", "wb") as f:
        pickle.dump(elo_snapshot, f)

    print(f"\nSaved: outcome_model.pkl, goals_model.pkl, elo_ratings.pkl → {MODEL_DIR}/")


if __name__ == "__main__":
    main()
