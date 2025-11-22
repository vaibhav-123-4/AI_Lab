#!/usr/bin/env python3
"""
lab5_hmm_2d.py
Gaussian HMM with 2D features for regime detection:
 - features: [log-return, rolling-volatility]
 - scaling with StandardScaler
 - GaussianHMM with full covariance
 - multi-restart per K with randomized initialization
 - model selection by BIC (fallback AIC)
 - saves plots, CSVs, and a text report

Usage: edit USER PARAMETERS below, then:
    python lab5_hmm_2d.py
"""
import os
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

# ---------------------------
# USER PARAMETERS
# ---------------------------
symbol = "AAPL"            # change as needed, e.g., "SPY", "^GSPC"
start_date = "2015-01-01"
end_date = None            # None -> today
min_k = 2
max_k = 5
n_restarts = 8             # more restarts -> more robust
random_seed = 2025
out_dir = "."
verbose = True
plot_dpi = 150
vol_window = 10            # rolling window for realized vol (days)
epsilon = 1e-6             # small value for numeric stability

# ---------------------------
# Setup
# ---------------------------
np.random.seed(random_seed)
sns.set(style="darkgrid", rc={"figure.dpi": plot_dpi})
plt.rcParams["figure.figsize"] = (10, 4)
os.makedirs(out_dir, exist_ok=True)
prefix = f"lab5_{symbol.replace('^','')}"
def outpath(name): return os.path.join(out_dir, f"{prefix}_{name}")

def msg(*a, **k):
    if verbose:
        print(*a, **k)

# ---------------------------
# 1. Download price data
# ---------------------------
msg("Downloading", symbol)
if end_date is None:
    end_date = datetime.today().strftime("%Y-%m-%d")
data = yf.download(symbol, start=start_date, end=end_date, progress=False)
if data.empty:
    raise RuntimeError("No data. Check symbol and dates.")
# choose Adjusted Close
if "Adj Close" in data.columns:
    prices = data["Adj Close"].dropna()
else:
    prices = data["Close"].dropna()
if len(prices) < 100:
    raise RuntimeError("Not enough data points. Use longer date range or different symbol.")
msg("Data range:", prices.index[0].date(), "to", prices.index[-1].date())
prices.to_csv(outpath("prices.csv"), index=True)

# ---------------------------
# 2. Compute features: returns and rolling vol
# ---------------------------
logret = np.log(prices).diff().dropna()
# realized volatility: rolling std of returns * sqrt(252?) Keep as daily vol (std of returns)
rolling_sd = logret.rolling(window=vol_window).std()
# align
df = pd.DataFrame({"ret": logret.values.flatten()}, index=logret.index)
df["vol"] = rolling_sd.values
df = df.dropna().copy()
df.reset_index(inplace=True)
df.rename(columns={"index": "date"}, inplace=True)
msg("Feature rows:", len(df))

# Save returns CSV
df.to_csv(outpath("returns_with_vol.csv"), index=False)

# ---------------------------
# 3. Feature matrix and scaling
# ---------------------------
X_raw = df[["ret", "vol"]].values  # shape (n_samples, 2)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ---------------------------
# 4. HMM utilities
# ---------------------------
def n_params_gaussian_hmm(K, D):
    # (K-1) startprob + K*(K-1) transmat + K*D means + K * D (diag vars)
    return (K - 1) + K * (K - 1) + K * D + K * D

# random init helper: set means near random data points, covars to sample covariance scaled
def random_init_hmm(hmm, X, rng):
    n, d = X.shape
    # choose K random points as means (in feature space)
    ids = rng.choice(n, size=hmm.n_components, replace=False)
    hmm.means_ = X[ids].copy()
    # covariances: start from sample covariance, make positive definite
    sample_cov = np.cov(X.T) + epsilon * np.eye(d)
    # set component covars slightly perturbed
    if hmm.covariance_type == "full":
        covs = np.array([sample_cov + rng.normal(scale=0.01, size=sample_cov.shape) for _ in range(hmm.n_components)])
        # ensure symmetry and positive definite
        for i in range(hmm.n_components):
            cov = (covs[i] + covs[i].T) / 2.0
            # add small diagonal if needed
            cov += epsilon * np.eye(d)
            covs[i] = cov
        hmm.covars_ = covs
    else:
        # diag
        var = np.diag(sample_cov)
        hmm.covars_ = np.tile(var + 0.001 * rng.rand(d), (hmm.n_components, 1))
    # init startprob uniform + tiny randomness
    sp = rng.rand(hmm.n_components)
    sp = sp / sp.sum()
    hmm.startprob_ = sp
    # init transmat stochastic
    tm = rng.rand(hmm.n_components, hmm.n_components)
    tm = (tm.T / tm.sum(axis=1)).T
    hmm.transmat_ = tm

# ---------------------------
# 5. Fit models for K range with restarts
# ---------------------------
results = []
models = {}
n_samples, n_features = X.shape

for K in range(min_k, max_k + 1):
    msg(f"\nFitting K={K} (restarts={n_restarts})")
    best_loglik = -np.inf
    best_model = None
    best_seed = None
    for r in range(n_restarts):
        seed_r = random_seed + r * 17 + K * 13
        rng = np.random.RandomState(seed_r)
        model = GaussianHMM(n_components=K, covariance_type="full", n_iter=300, verbose=False, random_state=seed_r)
        # random init: use means/covars/start/transition as reasonable starting points (makes EM escape degenerate solutions)
        try:
            # use partial-fit like hack: set attributes before fit (only works sometimes), but we'll fit and if fails try different init params
            random_init_hmm(model, X, rng)
            # call fit - hmmlearn will run EM from current params if attributes present
            model.fit(X)
        except Exception as e:
            # fallback: try plain fit (hmmlearn chooses its own init)
            try:
                model = GaussianHMM(n_components=K, covariance_type="full", n_iter=300, verbose=False, random_state=seed_r)
                model.fit(X)
            except Exception as e2:
                msg(f"  restart {r+1} failed: {e2}")
                continue
        # compute log-likelihood
        try:
            ll = model.score(X)  # log likelihood
        except Exception:
            ll = -np.inf
        msg(f"  restart {r+1}: seed={seed_r}, logLik={ll:.2f}")
        if ll > best_loglik:
            best_loglik = ll
            best_model = model
            best_seed = seed_r
    if best_model is None:
        msg(f"No successful fit for K={K}")
        continue
    # compute AIC/BIC
    npar = n_params_gaussian_hmm(K, n_features)
    aic = -2.0 * best_loglik + 2.0 * npar
    bic = -2.0 * best_loglik + npar * np.log(n_samples)
    results.append(dict(k=K, loglik=float(best_loglik), n_params=int(npar), AIC=float(aic), BIC=float(bic), seed=int(best_seed)))
    models[K] = best_model
    msg(f"Selected best restart seed={best_seed} for K={K}; logLik={best_loglik:.2f}; AIC={aic:.2f}; BIC={bic:.2f}")

# save model selection
res_df = pd.DataFrame(results).sort_values("k")
res_df.to_csv(outpath("model_selection.csv"), index=False)

# plot selection
plt.figure()
plt.plot(res_df['k'], res_df['AIC'], marker='o', label='AIC')
plt.plot(res_df['k'], res_df['BIC'], marker='o', label='BIC')
plt.plot(res_df['k'], res_df['loglik'], marker='o', label='logLik')
plt.xlabel("K (number of states)")
plt.legend()
plt.title("Model selection scores")
plt.savefig(outpath("model_selection.png"))
plt.close()

# ---------------------------
# 6. Choose best model (BIC) and decode
# ---------------------------
if res_df['BIC'].isnull().all():
    best_row = res_df.loc[res_df['AIC'].idxmin()]
    msg("BIC unavailable; using AIC")
else:
    best_row = res_df.loc[res_df['BIC'].idxmin()]
best_k = int(best_row['k'])
best_model = models[best_k]
msg("\nBest K selected:", best_k)

# decode
hidden = best_model.predict(X)
# posterior probabilities if available
post = None
try:
    post = best_model.predict_proba(X)
    has_post = True
    msg("Posterior probs computed.")
except Exception:
    has_post = False
    msg("predict_proba not available; skipping posterior plot.")

# attach to df (use original date index)
df["state"] = hidden
if has_post:
    for j in range(best_k):
        df[f"prob_S{j+1}"] = post[:, j]
df["date_num"] = np.arange(len(df))
df.to_csv(outpath("decoded_states.csv"), index=False)

# ---------------------------
# 7. State stats, emissions, transition
# ---------------------------
state_stats = df.groupby("state")["ret"].agg(["mean","std","median","count"]).reset_index().rename(columns={"std":"sd","count":"n"})
state_stats.to_csv(outpath("state_stats.csv"), index=False)

trans = best_model.transmat_
means = best_model.means_   # in scaled space
# convert means back to original feature scale
means_orig = scaler.inverse_transform(means)
covs = best_model.covars_   # full covariances (K, D, D)
# save emissions (convert back)
em_list = []
for i in range(best_k):
    m = means_orig[i]
    # convert cov to original scale: if X_scaled = (X - mu)/s, cov_orig = S @ cov_scaled @ S where S = diag(s)
    s = scaler.scale_
    S = np.diag(s)
    cov_scaled = covs[i]
    cov_orig = S.dot(cov_scaled).dot(S)
    em_list.append({"state": i, "mean_ret": float(m[0]), "mean_vol": float(m[1]), "sd_ret": float(np.sqrt(cov_orig[0,0])), "sd_vol": float(np.sqrt(cov_orig[1,1]))})
em_df = pd.DataFrame(em_list)
em_df.to_csv(outpath("emissions.csv"), index=False)
pd.DataFrame(trans).to_csv(outpath("transition_matrix.csv"), index=False)

# ---------------------------
# 8. Plots: returns colored by state, state sequence, posterior probs
# ---------------------------
palette = sns.color_palette("tab10", best_k)

# returns colored by state (scatter)
plt.figure(figsize=(12,4))
for s in range(best_k):
    mask = df['state'] == s
    plt.scatter(df.loc[mask, 'date_num'], df.loc[mask, 'ret'], s=8, color=palette[s], label=f"State {s}")
plt.legend()
plt.title(f"{symbol} returns colored by HMM state (K={best_k})")
plt.ylabel("log-return")
plt.xlabel("Time")
plt.savefig(outpath("returns_states.png"), dpi=plot_dpi, bbox_inches='tight')
plt.close()

# state sequence
plt.figure(figsize=(12,3))
plt.step(df['date_num'], df['state'], where='post')
plt.yticks(range(best_k))
plt.title("Decoded state sequence over time")
plt.xlabel("Time")
plt.ylabel("State")
plt.savefig(outpath("state_sequence.png"), dpi=plot_dpi, bbox_inches='tight')
plt.close()

# posterior probs
if has_post:
    plt.figure(figsize=(12,4))
    for j in range(best_k):
        plt.plot(df['date_num'], post[:, j], label=f"S{j}")
    plt.legend()
    plt.title("Posterior state probabilities")
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.savefig(outpath("posterior_probs.png"), dpi=plot_dpi, bbox_inches='tight')
    plt.close()

# price and returns plot
plt.figure(figsize=(12,4))
plt.plot(range(len(prices)), prices.values)
plt.title(f"{symbol} Adjusted Close")
plt.xlabel("Time")
plt.ylabel("Price")
plt.savefig(outpath("price.png"), dpi=plot_dpi, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12,3))
plt.plot(df['date_num'], df['ret'])
plt.title(f"{symbol} Log-Returns")
plt.xlabel("Time")
plt.ylabel("Log-Return")
plt.savefig(outpath("returns.png"), dpi=plot_dpi, bbox_inches='tight')
plt.close()

# ---------------------------
# 9. Text report
# ---------------------------
report = []
report.append("Gaussian HMM (2D features) Lab Report")
report.append("===============================")
report.append(f"Symbol: {symbol}")
report.append(f"Date range: {start_date} -> {end_date}")
report.append(f"Observations: {len(df)}")
report.append(f"Vol window: {vol_window} days")
report.append("")
report.append("Model selection (per K):")
report.append(res_df.to_string(index=False))
report.append("")
report.append("State stats (empirical returns):")
report.append(state_stats.to_string(index=False))
report.append("")
report.append("Emissions (means on original scale):")
report.append(em_df.to_string(index=False))
report.append("")
report.append("Transition matrix (rows=from, cols=to):")
report.append(pd.DataFrame(trans).to_string(index=False))
report.append("")
report.append("Files produced:")
files = sorted([f for f in os.listdir(out_dir) if f.startswith(prefix)])
report.extend(files)

with open(outpath("report.txt"), "w") as f:
    f.write("\n".join(map(str,report)))

msg("Done. Files (prefix):", prefix)
msg("\n".join(files))
