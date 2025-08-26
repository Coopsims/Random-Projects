"""
Synthetic time-series with seasonal sine signal (period=365) for 5 years,
plus an endogeneity demo (price -> demand) using:
  1) Naive model (biased)
  2) Control-Function (2SRI) with Random Forest
  3) Double ML (orthogonalization) for ATE
Includes both regression (continuous demand) and optional classification.

Author: you
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# -----------------------------
# 1) DATA GENERATION
# -----------------------------

@dataclass
class SimConfig:
    n_years: int = 5
    period: int = 365             # sine period
    noise_std: float = 0.7        # season noise
    seed: int = 42
    beta_price: float = -1.0      # true structural effect of price on demand (elasticity sign)
    endog_strength: float = 0.7   # correlates unobserved demand shock with price
    binary_outcome: bool = False  # set True to create a classification target


def make_time_index(cfg: SimConfig) -> pd.DatetimeIndex:
    n = cfg.n_years * cfg.period
    return pd.date_range("2018-01-01", periods=n, freq="D")


def make_sine_with_noise(n: int, period: int, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n)
    base = np.sin(2 * np.pi * t / period)
    # add a small 2nd harmonic for realism
    base += 0.3 * np.sin(4 * np.pi * t / period + 0.5)
    noise = rng.normal(0.0, noise_std, size=n)
    return base + noise


def simulate_endogenous_price_demand(cfg: SimConfig) -> pd.DataFrame:
    """
    Create a daily time series with seasonality (sine), exogenous drivers, endogenous price, and demand.
    Y = beta * Price + (exogenous season/inventory/etc) + u
    Price depends on exogenous factors + v, where v is correlated with u (endogeneity).
    """
    rng = np.random.default_rng(cfg.seed)
    idx = make_time_index(cfg)
    n = len(idx)

    # Core seasonal signal (period=365)
    season = make_sine_with_noise(n, cfg.period, cfg.noise_std, rng)

    # Exogenous drivers known at time t
    dow = pd.Series(idx.dayofweek, index=idx)                # 0..6
    month = pd.Series(idx.month, index=idx)                  # 1..12
    # Cyclical encodings (safe for trees but also fine)
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)
    mon_sin = np.sin(2 * np.pi * (month - 1) / 12.0)
    mon_cos = np.cos(2 * np.pi * (month - 1) / 12.0)

    # Inventory proxy (slow-moving)
    inventory = np.cumsum(rng.normal(0, 0.05, size=n)) + 10.0

    # Competitor price and cost shock (valid candidates for instruments/exogenous shifters)
    competitor_price = 20 + 2.0 * season + rng.normal(0, 0.8, size=n)
    cost_shock = rng.normal(0, 1.0, size=n)  # excluded from Y in structural equation (good IV candidate)

    # Unobserved demand shock u and an auxiliary eta to build v
    u = rng.normal(0, 1.0, size=n)
    eta = rng.normal(0, 1.0, size=n)
    v = cfg.endog_strength * u + np.sqrt(1 - cfg.endog_strength**2) * eta  # correlated with u

    # Build price_t (endogenous): depends on exogenous + v + small innovation
    price = (
        15
        + 1.8 * season
        + 0.6 * competitor_price
        + 0.9 * cost_shock
        + 0.05 * inventory
        + v
        + rng.normal(0, 0.5, size=n)
    )

    # Structural demand (continuous latent)
    y_latent = (
        cfg.beta_price * price
        + 3.0 * season
        - 0.03 * inventory
        + 0.5 * dow_sin + 0.5 * dow_cos
        + 0.3 * mon_sin + 0.3 * mon_cos
        + u
        + rng.normal(0, 0.5, size=n)
    )

    if cfg.binary_outcome:
        # Map latent to probability and sample 0/1
        p = 1 / (1 + np.exp(-y_latent))
        y = rng.binomial(1, np.clip(p, 1e-4, 1-1e-4))
    else:
        # Continuous demand; keep positive-ish by shifting
        y = y_latent + 10

    df = pd.DataFrame({
        "date": idx,
        "season": season,
        "dow": dow.values,
        "month": month.values,
        "dow_sin": dow_sin.values,
        "dow_cos": dow_cos.values,
        "mon_sin": mon_sin.values,
        "mon_cos": mon_cos.values,
        "inventory": inventory,
        "competitor_price": competitor_price,
        "cost_shock": cost_shock,
        "price": price,
        "y": y,
        "u": u,   # not observed in practice; kept for simulation diagnostics
        "v": v    # not observed in practice
    })
    # Lags (known at time t)
    df["price_lag1"] = df["price"].shift(1)
    df["y_lag1"] = df["y"].shift(1)
    df = df.dropna().reset_index(drop=True)
    return df


# -----------------------------
# 2) MODELING UTILITIES
# -----------------------------

def time_series_oof_predictions(model, X: pd.DataFrame, y: np.ndarray, n_splits: int = 5) -> Tuple[np.ndarray, list]:
    """
    Out-of-fold predictions with TimeSeriesSplit to avoid lookahead.
    Returns (oof_pred, list_of_fitted_fold_models)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.full(len(X), np.nan)
    models = []
    for train_idx, valid_idx in tscv.split(X):
        Xtr, Xva = X.iloc[train_idx], X.iloc[valid_idx]
        ytr = y[train_idx]
        mdl = clone_model(model)
        mdl.fit(Xtr, ytr)
        oof[valid_idx] = mdl.predict(Xva)
        models.append(mdl)
    return oof, models


def clone_model(model):
    # simple clone for RF/XGB regressors/classifiers without sklearn clone
    import copy
    return copy.deepcopy(model)


def train_test_split_time(df: pd.DataFrame, test_years: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_date = df["date"].max() - pd.Timedelta(days=365 * test_years - 1)
    tr = df[df["date"] < cutoff_date].copy()
    te = df[df["date"] >= cutoff_date].copy()
    return tr, te


# -----------------------------
# 3) NAIVE vs CONTROL-FUNCTION vs DML (REGRESSION)
# -----------------------------

def run_regression_pipeline(df: pd.DataFrame, use_xgb: bool = False) -> Dict[str, float]:
    """
    Compare:
      - Naive outcome model: y ~ price + controls
      - Control-Function (2SRI): Stage1 price ~ exogenous; Stage2 y ~ price + residual + controls
      - DML ATE: orthogonalized residual-on-residual regression for price effect
    Metrics: RMSE on test for outcome models; ATE estimates reported.
    """
    tr, te = train_test_split_time(df, test_years=1)

    # Features available at time t (no leakage)
    xw_cols = [
        "season", "inventory", "competitor_price", "cost_shock",
        "dow_sin", "dow_cos", "mon_sin", "mon_cos", "price_lag1", "y_lag1"
    ]
    y_tr = tr["y"].to_numpy()
    y_te = te["y"].to_numpy()

    # ---- Base learners
    rf_reg = RandomForestRegressor(n_estimators=600, min_samples_leaf=5, n_jobs=-1, random_state=1)
    if use_xgb and HAS_XGB:
        y_model = XGBRegressor(max_depth=6, n_estimators=500, subsample=0.9, colsample_bytree=0.9, random_state=1)
        t_model = XGBRegressor(max_depth=6, n_estimators=500, subsample=0.9, colsample_bytree=0.9, random_state=2)
    else:
        y_model = clone_model(rf_reg)
        t_model = clone_model(rf_reg)

    # -----------------
    # (A) Naive model
    # -----------------
    X_naive_tr = tr[["price"] + xw_cols]
    X_naive_te = te[["price"] + xw_cols]
    naive_model = clone_model(y_model)
    naive_model.fit(X_naive_tr, y_tr)
    naive_pred = naive_model.predict(X_naive_te)
    rmse_naive = mean_squared_error(y_te, naive_pred, squared=False)

    # -----------------
    # (B) Control-Function (2SRI)
    # Stage 1: price ~ X,W (out-of-fold on FULL df to create residuals without leakage)
    # -----------------
    X1 = df[xw_cols]
    t = df["price"].to_numpy()
    oof_price_hat, _ = time_series_oof_predictions(t_model, X1, t, n_splits=5)
    df_cf = df.copy()
    df_cf["price_hat_oof"] = oof_price_hat
    df_cf["price_residual"] = df_cf["price"] - df_cf["price_hat_oof"]

    tr_cf, te_cf = train_test_split_time(df_cf, test_years=1)
    X2_tr = tr_cf[["price", "price_residual"] + xw_cols]
    X2_te = te_cf[["price", "price_residual"] + xw_cols]

    cf_model = clone_model(y_model)
    cf_model.fit(X2_tr, tr_cf["y"].to_numpy())
    cf_pred = cf_model.predict(X2_te)
    rmse_cf = mean_squared_error(y_te, cf_pred, squared=False)

    # -----------------
    # (C) Double ML (ATE via orthogonalization / R-learner style)
    # -----------------
    # Nuisance: m(X,W) ~ E[Y|X,W] and g(X,W) ~ E[T|X,W], OOF via time splits
    m_hat_oof, _ = time_series_oof_predictions(y_model, df[xw_cols], df["y"].to_numpy(), n_splits=5)
    g_hat_oof, _ = time_series_oof_predictions(t_model, df[xw_cols], df["price"].to_numpy(), n_splits=5)

    df_dml = df.copy()
    df_dml["tilde_y"] = df_dml["y"] - m_hat_oof
    df_dml["tilde_t"] = df_dml["price"] - g_hat_oof
    # ATE: regress tilde_y on tilde_t with OLS (no intercept)
    mask = np.isfinite(df_dml["tilde_y"]) & np.isfinite(df_dml["tilde_t"])
    ate_model = LinearRegression(fit_intercept=False)
    ate_model.fit(df_dml.loc[mask, ["tilde_t"]], df_dml.loc[mask, "tilde_y"])
    ate = float(ate_model.coef_[0])

    return {
        "rmse_naive": rmse_naive,
        "rmse_control_function": rmse_cf,
        "dml_ate": ate
    }


# -----------------------------
# 4) CLASSIFICATION VARIANT (OPTIONAL)
# -----------------------------

def run_classification_pipeline(df: pd.DataFrame, use_xgb: bool = False) -> Dict[str, float]:
    """
    Binary Y version:
      - Naive classifier: y ~ price + controls
      - Control-Function classifier: y ~ price + price_residual + controls
      - DML ATE (logit-style score via residualization -> linear proxy for marginal effect)
    Metrics: AUC on test; report linear-proxy ATE for reference.
    """
    tr, te = train_test_split_time(df, test_years=1)

    xw_cols = [
        "season", "inventory", "competitor_price", "cost_shock",
        "dow_sin", "dow_cos", "mon_sin", "mon_cos", "price_lag1", "y_lag1"
    ]
    y_tr = tr["y"].astype(int).to_numpy()
    y_te = te["y"].astype(int).to_numpy()

    rf_clf = RandomForestClassifier(n_estimators=800, min_samples_leaf=10, n_jobs=-1, random_state=1, class_weight="balanced")
    if use_xgb and HAS_XGB:
        y_model = XGBClassifier(max_depth=6, n_estimators=600, subsample=0.9, colsample_bytree=0.9, random_state=1, eval_metric="auc")
        t_model = XGBRegressor(max_depth=6, n_estimators=500, subsample=0.9, colsample_bytree=0.9, random_state=2)
    else:
        y_model = clone_model(rf_clf)   # classifier for Y
        t_model = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=2)  # regressor for T

    # (A) Naive
    X_naive_tr = tr[["price"] + xw_cols]
    X_naive_te = te[["price"] + xw_cols]
    naive_clf = clone_model(y_model)
    naive_clf.fit(X_naive_tr, y_tr)
    if hasattr(naive_clf, "predict_proba"):
        p_naive = naive_clf.predict_proba(X_naive_te)[:, 1]
    else:
        p_naive = naive_clf.predict(X_naive_te)  # fallback
    auc_naive = roc_auc_score(y_te, p_naive)

    # (B) Control-Function residual inclusion (OOF residuals over full df)
    X1 = df[xw_cols]
    t = df["price"].to_numpy()
    oof_price_hat, _ = time_series_oof_predictions(t_model, X1, t, n_splits=5)
    df_cf = df.copy()
    df_cf["price_hat_oof"] = oof_price_hat
    df_cf["price_residual"] = df_cf["price"] - df_cf["price_hat_oof"]

    tr_cf, te_cf = train_test_split_time(df_cf, test_years=1)
    X2_tr = tr_cf[["price", "price_residual"] + xw_cols]
    X2_te = te_cf[["price", "price_residual"] + xw_cols]

    cf_clf = clone_model(y_model)
    cf_clf.fit(X2_tr, tr_cf["y"].astype(int).to_numpy())
    if hasattr(cf_clf, "predict_proba"):
        p_cf = cf_clf.predict_proba(X2_te)[:, 1]
    else:
        p_cf = cf_clf.predict(X2_te)
    auc_cf = roc_auc_score(y_te, p_cf)

    # (C) DML proxy ATE (linear probability approximation on orthogonalized residuals)
    m_hat_oof, _ = time_series_oof_predictions(y_model if HAS_XGB else rf_clf, df[xw_cols], df["y"].astype(int).to_numpy(), n_splits=5)
    # convert proba -> expectation if classifier
    if m_hat_oof.ndim == 1:
        m_hat = m_hat_oof
    else:
        m_hat = m_hat_oof  # already vector of probs for XGB; keep as is

    g_hat_oof, _ = time_series_oof_predictions(RandomForestRegressor(n_estimators=600, min_samples_leaf=5, n_jobs=-1, random_state=3),
                                               df[xw_cols], df["price"].to_numpy(), n_splits=5)

    df_dml = df.copy()
    df_dml["tilde_y"] = df_dml["y"].astype(int).to_numpy() - m_hat
    df_dml["tilde_t"] = df_dml["price"] - g_hat_oof
    mask = np.isfinite(df_dml["tilde_y"]) & np.isfinite(df_dml["tilde_t"])

    ate_proxy = LinearRegression(fit_intercept=False).fit(
        df_dml.loc[mask, ["tilde_t"]], df_dml.loc[mask, "tilde_y"]
    ).coef_[0]

    return {
        "auc_naive": auc_naive,
        "auc_control_function": auc_cf,
        "dml_ate_proxy": float(ate_proxy)
    }


# -----------------------------
# 5) MAIN
# -----------------------------

if __name__ == "__main__":
    # --- CONFIGURE SIM ---
    cfg = SimConfig(
        n_years=5,
        period=365,
        noise_std=0.7,
        seed=123,
        beta_price=-1.2,       # true structural price effect (negative)
        endog_strength=0.7,
        binary_outcome=False   # set True to switch to classification variant
    )

    df = simulate_endogenous_price_demand(cfg)

    if not cfg.binary_outcome:
        results = run_regression_pipeline(df, use_xgb=True)
        print("\n=== REGRESSION RESULTS (period=365, 5 years) ===")
        for k, v in results.items():
            print(f"{k:>28s}: {v: .4f}")
        print("Note: dml_ate should be close to the true beta_price:", cfg.beta_price)
    else:
        # turn y into 0/1 if not already
        df["y"] = (df["y"] > 0).astype(int)
        results = run_classification_pipeline(df, use_xgb=True)
        print("\n=== CLASSIFICATION RESULTS (period=365, 5 years) ===")
        for k, v in results.items():
            print(f"{k:>28s}: {v: .4f}")