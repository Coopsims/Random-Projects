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

# Plotting
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts/servers
import matplotlib.pyplot as plt

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
      - DMLIV ATE: IV-based DML using excluded instruments (Z) to identify the causal effect
    Metrics: RMSE on test for outcome models; ATE estimates reported.
    """
    tr, te = train_test_split_time(df, test_years=1)

    # Features available at time t (no leakage)
    xw_cols = [
        "season", "inventory", "competitor_price", "cost_shock",
        "dow_sin", "dow_cos", "mon_sin", "mon_cos", "price_lag1", "y_lag1"
    ]
    # For DMLIV, separate controls X and instruments Z (excluded from Y equation)
    z_cols = ["cost_shock", "competitor_price"]
    x_cols = [c for c in xw_cols if c not in z_cols]

    y_tr = tr["y"].to_numpy()
    y_te = te["y"].to_numpy()

    # ---- Base learners
    rf_reg = RandomForestRegressor(n_estimators=600, min_samples_leaf=5, n_jobs=-1, random_state=1)
    if use_xgb and HAS_XGB:
        y_model = XGBRegressor(max_depth=6, n_estimators=500, subsample=0.9, colsample_bytree=0.9, random_state=1, n_jobs=-1)
        t_model = XGBRegressor(max_depth=6, n_estimators=500, subsample=0.9, colsample_bytree=0.9, random_state=2, n_jobs=-1)
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
    rmse_naive = float(np.sqrt(mean_squared_error(y_te, naive_pred)))

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
    rmse_cf = float(np.sqrt(mean_squared_error(y_te, cf_pred)))

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

    # -----------------
    # (D) DMLIV (IV-based DML using excluded instruments Z)
    # -----------------
    # m(X) = E[Y|X]
    m_iv_oof, _ = time_series_oof_predictions(y_model, df[x_cols], df["y"].to_numpy(), n_splits=5)
    # g(X) = E[T|X]
    g_x_oof, _ = time_series_oof_predictions(t_model, df[x_cols], df["price"].to_numpy(), n_splits=5)
    # r(X,Z) = E[T|X,Z]
    r_xz_oof, _ = time_series_oof_predictions(t_model, df[x_cols + z_cols], df["price"].to_numpy(), n_splits=5)

    df_iv = df.copy()
    df_iv["tilde_y_iv"] = df_iv["y"] - m_iv_oof
    df_iv["w_hat"] = r_xz_oof - g_x_oof

    mask_iv = np.isfinite(df_iv["tilde_y_iv"]) & np.isfinite(df_iv["w_hat"]) & (np.abs(df_iv["w_hat"]) > 1e-8)
    dmliv_model = LinearRegression(fit_intercept=False)
    dmliv_model.fit(df_iv.loc[mask_iv, ["w_hat"]], df_iv.loc[mask_iv, "tilde_y_iv"])
    dmliv_ate = float(dmliv_model.coef_[0])

    return {
        "rmse_naive": rmse_naive,
        "rmse_control_function": rmse_cf,
        "dml_ate": ate,
        "dmliv_ate": dmliv_ate
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
# 5) PLOTTING HELPERS
# -----------------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _add_trend(ax, x, y, color, label):
    try:
        lr = LinearRegression().fit(x.reshape(-1, 1), y)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        ax.plot(xs, lr.predict(xs.reshape(-1, 1)), color=color, lw=2, label=label)
    except Exception:
        pass


def generate_plots(df: pd.DataFrame, results: Dict[str, float], cfg: SimConfig, outdir: str = "causal-modeling/figures") -> Dict[str, str]:
    """
    Generate and save a set of illustrative figures explaining the simulation and estimators.
    Returns a dict of figure names to file paths.
    """
    _ensure_dir(outdir)
    paths = {}

    # 1) Time series of price and demand (last ~2 years for readability)
    try:
        cutoff = df["date"].max() - pd.Timedelta(days=365*2 - 1)
        dfx = df[df["date"] >= cutoff].copy()
        fig, ax1 = plt.subplots(figsize=(11, 5))
        ax1.plot(dfx["date"], dfx["y"], color="#1f77b4", label="Demand (y)")
        ax1.set_ylabel("Demand (y)", color="#1f77b4")
        ax1.tick_params(axis='y', labelcolor="#1f77b4")
        ax2 = ax1.twinx()
        ax2.plot(dfx["date"], dfx["price"], color="#ff7f0e", alpha=0.8, label="Price")
        ax2.set_ylabel("Price", color="#ff7f0e")
        ax2.tick_params(axis='y', labelcolor="#ff7f0e")
        ax1.set_title("Time series: Demand and Price (last 2 years)")
        fig.autofmt_xdate()
        fig.tight_layout()
        p = os.path.join(outdir, "timeseries_price_demand.png")
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths["timeseries"] = p
    except Exception:
        pass

    # 2) Scatter y vs price + slope lines (true beta, DML ATE, naive simple)
    try:
        x = df["price"].to_numpy()
        y = df["y"].to_numpy()
        mx, my = np.nanmean(x), np.nanmean(y)
        fig, ax = plt.subplots(figsize=(7.5, 6))
        ax.scatter(x, y, s=8, alpha=0.25, color="#333333")
        # Lines through mean point to visualize slope
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        # True beta line
        ax.plot(xs, my + cfg.beta_price * (xs - mx), color="red", lw=2, label=f"True beta={cfg.beta_price:.2f}")
        # DML ATE line (from results)
        ate = results.get("dml_ate") or results.get("dml_ate_proxy")
        if ate is not None:
            ax.plot(xs, my + ate * (xs - mx), color="green", lw=2, label=f"DML ATE={ate:.2f}")
        # DMLIV ATE line (IV-based)
        ate_iv = results.get("dmliv_ate")
        if ate_iv is not None:
            ax.plot(xs, my + ate_iv * (xs - mx), color="#7f7fff", lw=2, label=f"DMLIV ATE={ate_iv:.2f}")
        # Naive simple OLS (y ~ price)
        try:
            naive_beta = LinearRegression().fit(x.reshape(-1, 1), y).coef_[0]
            ax.plot(xs, my + naive_beta * (xs - mx), color="blue", lw=2, linestyle="--", label=f"Naive slope={naive_beta:.2f}")
        except Exception:
            pass
        ax.set_xlabel("Price")
        ax.set_ylabel("Demand (y)")
        ax.set_title("Demand vs Price with slopes (True vs Estimated)")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        p = os.path.join(outdir, "scatter_y_vs_price_slopes.png")
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths["y_vs_price_slopes"] = p
    except Exception:
        pass

    # 3) Instrument relevance: price vs cost_shock and competitor_price
    try:
        for col, fname in [("cost_shock", "price_vs_cost_shock.png"), ("competitor_price", "price_vs_competitor_price.png")]:
            fig, ax = plt.subplots(figsize=(7, 5))
            xv = df[col].to_numpy()
            yv = df["price"].to_numpy()
            ax.scatter(xv, yv, s=8, alpha=0.25, color="#2ca02c")
            _add_trend(ax, xv, yv, color="#d62728", label="Linear trend")
            ax.set_xlabel(col)
            ax.set_ylabel("Price")
            ax.set_title(f"Instrument relevance: Price vs {col}")
            ax.legend()
            ax.grid(alpha=0.2)
            fig.tight_layout()
            p = os.path.join(outdir, fname)
            fig.savefig(p, dpi=160)
            plt.close(fig)
            paths[col] = p
    except Exception:
        pass

    # 4) Endogeneity diagnostic: u vs v and u vs price (only available in simulation)
    try:
        if "u" in df.columns and "v" in df.columns:
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            ax.scatter(df["u"], df["v"], s=8, alpha=0.25, color="#9467bd")
            ax.set_xlabel("Unobserved demand shock (u)")
            ax.set_ylabel("Price shock component (v)")
            ax.set_title("Endogeneity: Correlated shocks u and v")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            p = os.path.join(outdir, "u_vs_v.png")
            fig.savefig(p, dpi=160)
            plt.close(fig)
            paths["u_vs_v"] = p

            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            ax.scatter(df["u"], df["price"], s=8, alpha=0.25, color="#8c564b")
            _add_trend(ax, df["u"].to_numpy(), df["price"].to_numpy(), color="#e377c2", label="Linear trend")
            ax.set_xlabel("Unobserved demand shock (u)")
            ax.set_ylabel("Price")
            ax.set_title("Resulting correlation: Price vs u (endogeneity)")
            ax.legend()
            ax.grid(alpha=0.2)
            fig.tight_layout()
            p = os.path.join(outdir, "price_vs_u.png")
            fig.savefig(p, dpi=160)
            plt.close(fig)
            paths["price_vs_u"] = p
    except Exception:
        pass

    # 5) Season confounding view: season vs price
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(df["season"], df["price"], s=8, alpha=0.2, color="#17becf")
        _add_trend(ax, df["season"].to_numpy(), df["price"].to_numpy(), color="#bcbd22", label="Linear trend")
        ax.set_xlabel("Season signal")
        ax.set_ylabel("Price")
        ax.set_title("Seasonal driver relation: Price vs Season")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        p = os.path.join(outdir, "price_vs_season.png")
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths["price_vs_season"] = p
    except Exception:
        pass

    return paths


# -----------------------------
# 6) MAIN
# -----------------------------

if __name__ == "__main__":
    # --- CONFIGURE SIM ---
    cfg = SimConfig(
        n_years=5,
        period=365,
        noise_std=0.3,
        seed=17,
        beta_price=-1.2,       # true structural price effect (negative)
        endog_strength=0.7,
        binary_outcome=False   # set True to switch to classification variant
    )

    df = simulate_endogenous_price_demand(cfg)

    if not cfg.binary_outcome:
        results = run_regression_pipeline(df, use_xgb=False)
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

    # Generate and report plots
    try:
        fig_paths = generate_plots(df, results, cfg, outdir="causal-modeling/figures")
        if fig_paths:
            print("\nSaved figures:")
            for name, p in fig_paths.items():
                print(f" - {name}: {p}")
        else:
            print("\nNo figures were generated.")
    except Exception as e:
        print("Plotting failed:", e)