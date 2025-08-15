# ===============================
# Interactive Portfolio Optimizer Dashboard
# Investor-focused with Methods: Monte Carlo, Linear Regression, Correlation & Multicollinearity, PCA
# ===============================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from datetime import datetime

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="Interactive Portfolio Optimizer", layout="wide")

# ---------------------------------
# SIDEBAR – CONTROLS
# ---------------------------------
st.sidebar.title("⚙️ Controls")

def _default_tickers():
    return "MSFT,AAPL,AMZN,TSLA,GOOGL"

def _load_prices(tickers, benchmark, start, end):
    data = yf.download(tickers + [benchmark], start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        px = data["Close"].copy()
    else:
        px = data.copy()
    return px.dropna(how="all")

with st.sidebar:
    tickers_str = st.text_input("Tickers (comma-separated)", value=_default_tickers())
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    benchmark = st.text_input("Benchmark (Yahoo symbol)", value="^GSPC")
    colA, colB = st.columns(2)
    with colA:
        start_date = st.date_input("Start", value=pd.to_datetime("2018-01-01")).strftime("%Y-%m-%d")
    with colB:
        end_date = st.date_input("End", value=pd.to_datetime("today")).strftime("%Y-%m-%d")

    st.markdown("---")
    st.subheader("Portfolio Weights")
    raw_weights = []
    for t in tickers:
        raw_weights.append(st.slider(f"{t}", 0.0, 1.0, 1.0/max(len(tickers),1), 0.01))
    weights = np.array(raw_weights)
    if weights.sum() == 0:
        weights = np.array([1/len(tickers)]*len(tickers))
    weights = weights / weights.sum()

    st.markdown("---")
    st.subheader("Simulation Settings")
    init_cap = st.number_input("Initial Capital", value=100000, step=1000)
    n_sims = st.slider("Monte Carlo Simulations", 500, 20000, 5000, 500)
    horizon_days = st.slider("Horizon (trading days)", 63, 756, 252, 21)
    shock = st.selectbox("Scenario Shock", ["None", "-10% Day 1", "+10% Day 1", "Volatility x1.5"])    

    st.markdown("---")
    st.subheader("Optimizer")
    do_opt = st.checkbox("Optimize for max Sharpe (no shorting)")
    opt_trials = st.slider("Search Trials", 1000, 20000, 5000, 500)
    rf_rate = st.number_input("Risk-free (annual)", value=0.02, step=0.005, format="%.3f")

# ---------------------------------
# DATA
# ---------------------------------
px = _load_prices(tickers, benchmark, start_date, end_date)
missing = [t for t in tickers+[benchmark] if t not in px.columns]
if missing:
    st.warning(f"Missing data for: {', '.join(missing)}. Check symbols/date range.")

# Keep only available
tickers = [t for t in tickers if t in px.columns]
if len(tickers) == 0:
    st.stop()

rets = px.pct_change().dropna()
bench_rets = rets[benchmark].copy() if benchmark in rets.columns else None
asset_rets = rets[tickers].copy()

# Align weights
if len(weights) != len(tickers):
    weights = np.array([1/len(tickers)]*len(tickers))

# ---------------------------------
# FUNCTIONS
# ---------------------------------
TRADING_DAYS = 252

def portfolio_series(asset_returns, w):
    return (asset_returns @ w).rename("Portfolio")

pf_rets = portfolio_series(asset_rets, weights)
pf_cum = (1+pf_rets).cumprod()
bench_cum = (1+bench_rets).cumprod() if bench_rets is not None else None

# Metrics
def ann_return(series):
    return (1+series.mean())**TRADING_DAYS - 1

def ann_vol(series):
    return series.std()*np.sqrt(TRADING_DAYS)

def sharpe(series, rf=0.0):
    return (ann_return(series)-rf)/ann_vol(series) if ann_vol(series) > 0 else np.nan

def max_dd(cum):
    return (cum/cum.cummax()-1).min()

ann_ret = ann_return(pf_rets)
ann_vol_ = ann_vol(pf_rets)
sharpe_ = sharpe(pf_rets, rf_rate)
dd = max_dd(pf_cum)

# ---------------------------------
# MONTE CARLO (correlated, multivariate normal)
# ---------------------------------
mu = asset_rets.mean().values  # daily mean
cov = asset_rets.cov().values  # daily cov

# Apply scenario shock
shock_mu = mu.copy()
shock_cov = cov.copy()
if shock == "Volatility x1.5":
    shock_cov = shock_cov * (1.5**2)

# Simulate
rng = np.random.default_rng(42)
L = np.linalg.cholesky(shock_cov + 1e-12*np.eye(len(tickers)))
Z = rng.standard_normal(size=(n_sims, horizon_days, len(tickers)))
shock_day = np.zeros((n_sims, len(tickers)))
if shock in ["-10% Day 1", "+10% Day 1"]:
    shock_day = (-0.10 if "-10%" in shock else 0.10) * np.ones((n_sims, len(tickers)))

# Daily returns path: r_t = mu + L*Z
paths = mu + (Z @ L.T)
# inject day-1 shock (t=0)
paths[:,0,:] = paths[:,0,:] + shock_day

# Convert to portfolio paths
pf_daily_paths = (paths @ weights)
pf_value_paths = init_cap * np.cumprod(1 + pf_daily_paths, axis=1)
terminal_values = pf_value_paths[:,-1]

mc_mean = np.mean(terminal_values)
mc_p5 = np.percentile(terminal_values, 5)
mc_p50 = np.percentile(terminal_values, 50)
mc_p95 = np.percentile(terminal_values, 95)

# VaR/CVaR (1-day, from historical pf_rets)
alpha = 0.05
VaR = norm.ppf(alpha, loc=pf_rets.mean(), scale=pf_rets.std())
CVaR = pf_rets[pf_rets <= VaR].mean() if (pf_rets <= VaR).any() else np.nan

# ---------------------------------
# OPTIONAL: SIMPLE SHARPE OPTIMIZER (random search, long-only)
# ---------------------------------
opt_weights = None
opt_metrics = None
if do_opt:
    best_s, best_w = -1e9, None
    # Dirichlet samples ensure sum=1 and w>=0
    for _ in range(opt_trials):
        w = rng.dirichlet(np.ones(len(tickers)))
        s = sharpe(portfolio_series(asset_rets, w), rf_rate)
        if s > best_s:
            best_s, best_w = s, w
    opt_weights = best_w
    opt_metrics = {
        "Ann Return": ann_return(portfolio_series(asset_rets, opt_weights)),
        "Ann Vol": ann_vol(portfolio_series(asset_rets, opt_weights)),
        "Sharpe": best_s,
    }

# ---------------------------------
# HEADER
# ---------------------------------
st.title("📈 Interactive Portfolio Optimizer Dashboard")
st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# KPI Row
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Initial Capital", f"${init_cap:,.0f}")
k2.metric("Current Value (Backtest)", f"${(init_cap*pf_cum.iloc[-1]):,.0f}")
k3.metric("Total Return", f"{(pf_cum.iloc[-1]-1)*100:.2f}%")
k4.metric("Annualized Return", f"{ann_ret*100:.2f}%")
k5.metric("Sharpe (ann)", f"{sharpe_: .2f}")
k6.metric("Max Drawdown", f"{dd:.2%}")

# Key Takeaways
bullets = []
if sharpe_ > 1: bullets.append("Risk-adjusted performance is solid (Sharpe > 1).")
if bench_cum is not None and pf_cum.iloc[-1] > bench_cum.iloc[-1]:
    bullets.append("Portfolio outperformed benchmark over the selected period.")
if dd < -0.3: bullets.append("Significant historical drawdown observed; consider diversification.")
if VaR < -0.02: bullets.append(f"1-day 5% VaR ≈ {VaR:.2%} implies notable short-term downside risk.")
if do_opt and opt_weights is not None: bullets.append("Optimizer found a higher-Sharpe mix (see Optimizer panel).")

with st.expander("Key Takeaways (auto-generated)", expanded=True):
    for b in bullets:
        st.write(f"- {b}")

# ---------------------------------
# TABS
# ---------------------------------

tab_overview, tab_risk, tab_sim, tab_methods, tab_opt = st.tabs([
    "Overview",
    "Risk",
    "Monte Carlo",
    "Methods: Regression • Correlation • PCA • VIF",
    "Optimizer",
])

# ---- Overview ----
with tab_overview:
    st.subheader("Cumulative Performance")
    fig, ax = plt.subplots()
    ax.plot(pf_cum.index, pf_cum.values, label="Portfolio")
    if bench_cum is not None:
        ax.plot(bench_cum.index, bench_cum.values, label="Benchmark")
    ax.set_title("Cumulative Returns (Normalized)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Allocation")
    alloc_df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
    st.dataframe(alloc_df.style.format({"Weight": "{:.2%}"}), use_container_width=True)

# ---- Risk ----
with tab_risk:
    c1, c2, c3 = st.columns(3)
    c1.metric("Annualized Volatility", f"{ann_vol_: .2%}")
    c2.metric("1-day VaR (5%)", f"{VaR: .2%}")
    c3.metric("1-day CVaR (5%)", f"{CVaR: .2%}" if not np.isnan(CVaR) else "n/a")

    st.subheader("Drawdown")
    dd_series = pf_cum/pf_cum.cummax() - 1
    fig, ax = plt.subplots()
    ax.plot(dd_series.index, dd_series.values)
    ax.set_title("Drawdown History")
    st.pyplot(fig)

    st.subheader("Risk/Return Scatter vs Components")
    rr = pd.DataFrame({
        "Ann Return": asset_rets.apply(ann_return),
        "Ann Vol": asset_rets.apply(ann_vol),
    })
    rr.loc["Portfolio"] = [ann_ret, ann_vol_]
    fig, ax = plt.subplots()
    ax.scatter(rr["Ann Vol"], rr["Ann Return"])
    for lbl, row in rr.iterrows():
        ax.annotate(lbl, (row["Ann Vol"], row["Ann Return"]))
    ax.set_xlabel("Volatility (ann)")
    ax.set_ylabel("Return (ann)")
    ax.set_title("Risk vs Return")
    st.pyplot(fig)

# ---- Monte Carlo ----
with tab_sim:
    st.subheader("Terminal Value Distribution")
    fig, ax = plt.subplots()
    ax.hist(terminal_values, bins=50)
    ax.axvline(mc_p5, linestyle='--', label="5%")
    ax.axvline(mc_p50, linestyle='--', label="50% (Median)")
    ax.axvline(mc_p95, linestyle='--', label="95%")
    ax.set_title(f"Monte Carlo Terminal Values (n={n_sims}, horizon={horizon_days}d)")
    ax.legend()
    st.pyplot(fig)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("P5", f"${mc_p5:,.0f}")
    c2.metric("Median", f"${mc_p50:,.0f}")
    c3.metric("Mean", f"${mc_mean:,.0f}")
    c4.metric("P95", f"${mc_p95:,.0f}")

# ---- Methods: Regression • Correlation • PCA • VIF ----
with tab_methods:
    st.markdown("### Correlation Matrix")
    corr = asset_rets.corr()
    st.dataframe(corr.style.background_gradient(cmap="RdBu", axis=None).format("{:.2f}"), use_container_width=True)

    st.markdown("### Linear Regression (CAPM-style): Asset vs Benchmark")
    if bench_rets is not None:
        rows = []
        X = sm.add_constant(bench_rets.values)
        for t in tickers:
            y = asset_rets[t].values
            model = sm.OLS(y, X).fit()
            alpha, beta = model.params[0], model.params[1]
            rows.append({"Ticker": t, "Alpha (const)": alpha, "Beta": beta, "R²": model.rsquared})
        # Portfolio vs Benchmark
        model_pf = sm.OLS(pf_rets.values, X).fit()
        rows.append({"Ticker": "Portfolio", "Alpha (const)": model_pf.params[0], "Beta": model_pf.params[1], "R²": model_pf.rsquared})
        reg_df = pd.DataFrame(rows)
        st.dataframe(reg_df.style.format({"Alpha (const)": "{:.5f}", "Beta": "{:.2f}", "R²": "{:.2f}"}), use_container_width=True)
    else:
        st.info("Benchmark not available for regression.")

    st.markdown("### Multicollinearity (VIF) among Assets")
    # VIF expects a feature matrix; use asset returns (standardized)
    Xv = (asset_rets - asset_rets.mean())/asset_rets.std()
    Xv = Xv.dropna()
    vif_rows = []
    try:
        for i, col in enumerate(Xv.columns):
            vif_rows.append({"Variable": col, "VIF": variance_inflation_factor(Xv.values, i)})
        vif_df = pd.DataFrame(vif_rows)
        st.dataframe(vif_df.style.format({"VIF": "{:.2f}"}), use_container_width=True)
        st.caption("Rule of thumb: VIF > 5–10 suggests high multicollinearity.")
    except Exception as e:
        st.warning(f"VIF could not be computed: {e}")

    st.markdown("### Principal Component Analysis (PCA)")
    pca = PCA()
    pca.fit(Xv.values)
    explained = pd.Series(pca.explained_variance_ratio_, name="Explained Var Ratio")
    loadings = pd.DataFrame(pca.components_.T, index=Xv.columns)
    loadings.columns = [f"PC{i+1}" for i in range(loadings.shape[1])]

    c1, c2 = st.columns([1,1])
    with c1:
        st.write("Explained Variance Ratio")
        st.dataframe(explained.head(10).to_frame().style.format({"Explained Var Ratio": "{:.2%}"}), use_container_width=True)
    with c2:
        st.write("First 3 PCs – Loadings")
        st.dataframe(loadings.iloc[:, :min(3, loadings.shape[1])].style.format("{:.2f}"), use_container_width=True)

# ---- Optimizer ----
with tab_opt:
    st.subheader("Random-Search Max Sharpe (Long-only)")
    if not do_opt or opt_weights is None:
        st.info("Tick the 'Optimize for max Sharpe' checkbox in the sidebar and set trials to search.")
    else:
        st.success("Optimizer completed – results below.")
        opt_alloc = pd.DataFrame({"Ticker": tickers, "Weight": opt_weights})
        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpe (ann)", f"{opt_metrics['Sharpe']:.2f}")
        c2.metric("Ann Return", f"{opt_metrics['Ann Return']*100:.2f}%")
        c3.metric("Ann Vol", f"{opt_metrics['Ann Vol']*100:.2f}%")
        st.dataframe(opt_alloc.style.format({"Weight": "{:.2%}"}), use_container_width=True)

        # Compare current vs optimized allocations
        st.subheader("Current vs Optimized Allocation")
        cmp = pd.DataFrame({
            "Current": weights,
            "Optimized": opt_weights
        }, index=tickers)
        st.dataframe(cmp.style.format("{:.2%}"), use_container_width=True)

        # Plot cumulative vs benchmark for optimized
        opt_pf = portfolio_series(asset_rets, opt_weights)
        opt_cum = (1+opt_pf).cumprod()
        fig, ax = plt.subplots()
        ax.plot(pf_cum.index, pf_cum.values, label="Current")
        ax.plot(opt_cum.index, opt_cum.values, label="Optimized")
        if bench_cum is not None:
            ax.plot(bench_cum.index, bench_cum.values, label="Benchmark")
        ax.set_title("Cumulative Performance: Current vs Optimized")
        ax.legend()
        st.pyplot(fig)

# ---------------------------------
# FOOTER NOTES
# ---------------------------------
st.caption("This dashboard is for educational purposes, demonstrating interactive portfolio analysis with Monte Carlo simulations, CAPM-style regression, correlation & multicollinearity diagnostics (VIF), and PCA. Not investment advice.")
