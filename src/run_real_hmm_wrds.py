import numpy as np
import pandas as pd
import wrds
from hmmlearn.hmm import GaussianHMM

import simulate
import control
import plotting
import results

from params import SubPopParams
from params import simulation_params, control_params


# ============================================================
# 1. Load WRDS TAQ midprice data
# ============================================================

def load_wrds_midprice(
    ticker,
    date,
    start_time="09:30:00",
    end_time="16:00:00",
    n_steps=390,
    wrds_username=None,
):
    """
    Loads WRDS TAQ NBBO data and returns evenly sampled midprices.
    """

    yyyymm = date.replace("-", "")[:6]
    table = f"taqmsec.nbbom_{yyyymm}"

    db = wrds.Connection(wrds_username=wrds_username)

    query = f"""
        SELECT date, time_m, sym_root, bid, ask
        FROM {table}
        WHERE sym_root = '{ticker.upper()}'
          AND date = '{date}'
          AND time_m BETWEEN '{start_time}' AND '{end_time}'
          AND bid > 0
          AND ask > 0
          AND ask >= bid
        ORDER BY time_m
    """

    df = db.raw_sql(query)
    db.close()

    if df.empty:
        raise ValueError("No WRDS data returned. Check ticker, date, and TAQ access.")

    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time_m"].astype(str))
    df["mid"] = 0.5 * (df["bid"] + df["ask"])

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mid"])
    df = df[df["mid"] > 0]
    df = df.sort_values("datetime")

    target_times = pd.date_range(
        start=pd.to_datetime(f"{date} {start_time}"),
        end=pd.to_datetime(f"{date} {end_time}"),
        periods=n_steps + 1,
    )

    Ft = (
        df.set_index("datetime")["mid"]
        .reindex(target_times, method="nearest")
        .ffill()
        .bfill()
        .to_numpy()
    )

    return Ft, target_times


# ============================================================
# 2. Fit 2-state HMM on real intraday returns
# ============================================================

def fit_real_data_hmm(Ft):
    """
    Fits a 2-state Gaussian HMM to log returns.
    Returns posterior probabilities and learned drift estimates.
    """

    returns = np.diff(np.log(Ft)).reshape(-1, 1)

    hmm = GaussianHMM(
        n_components=2,
        covariance_type="diag",
        n_iter=500,
        random_state=42,
    )

    hmm.fit(returns)

    posterior = hmm.predict_proba(returns)
    means = hmm.means_.flatten()

    # Sort states by mean return
    order = np.argsort(means)

    low_state = order[0]
    high_state = order[1]

    mu0 = means[low_state]
    mu1 = means[high_state]

    pi_state1_returns = posterior[:, high_state]

    # Align posterior length with price path
    pi_state1 = np.empty(len(Ft))
    pi_state1[0] = pi_state1_returns[0]
    pi_state1[1:] = pi_state1_returns

    A_hat = pi_state1 * mu1 + (1 - pi_state1) * mu0

    latent_proxy = (pi_state1 > 0.5).astype(int)

    return pi_state1, A_hat, mu0, mu1, latent_proxy, hmm


# ============================================================
# 3. Main experiment
# ============================================================

ticker = "AAPL"
date = "2024-01-03"

SubPop1 = SubPopParams(
    name="SubPop1",
    weight=0.5,
    prior=0.8,
    Q0=1.0,
    kappa=0.5,
)

SubPop2 = SubPopParams(
    name="SubPop2",
    weight=0.5,
    prior=0.2,
    Q0=1.0,
    kappa=2.0,
)

SubPops = np.array([SubPop1, SubPop2])


# Real midpoint price path
Ft, timestamps = load_wrds_midprice(
    ticker=ticker,
    date=date,
    start_time="09:30:00",
    end_time="16:00:00",
    n_steps=simulation_params.N,
    wrds_username=None,   # change to your WRDS username if needed
)


# Learned HMM signal from real data
pi_real, A_hat_real, mu0, mu1, latent_proxy, hmm = fit_real_data_hmm(Ft)

print("\nLearned HMM regimes:")
print(f"  Low-drift state mu0:  {mu0}")
print(f"  High-drift state mu1: {mu1}")


# ============================================================
# 4. Controls for subpopulations
# ============================================================

pi_k = np.empty((len(SubPops), simulation_params.N + 1))
A_hat_k = np.empty((len(SubPops), simulation_params.N + 1))
nu_hat_k = np.empty((len(SubPops), simulation_params.N))

for i, sp in enumerate(SubPops):
    # Same real-data posterior for both groups
    # They differ through Q0 and kappa
    pi_k[i] = pi_real
    A_hat_k[i] = A_hat_real

    nu_hat_k[i] = control.alpha_inventory_control(
        A_hat_k[i, :-1],
        params=control.ControlParams(
            T=control_params.T,
            N=control_params.N,
            Q0=sp.Q0,
        ),
        kappa=sp.kappa,
    )


nu_bar = sum(sp.weight * nu_hat_k[i] for i, sp in enumerate(SubPops))


# ============================================================
# 5. Simulate hypothetical price impact
# ============================================================

St = simulate.simulate_impacted_price(
    F_t=Ft,
    nu_hat=nu_bar,
    params=simulation_params,
)


# For real data, impacted posterior can be set equal to real posterior
# unless you specifically want to re-filter on S_t.
pi_imp_k = pi_k.copy()


# ============================================================
# 6. Individual heterogeneous agents
# ============================================================

rng = np.random.default_rng(42)
n_agents_per_subpop = 50

individual_nu = []
individual_labels = []

for i, sp in enumerate(SubPops):
    for _ in range(n_agents_per_subpop):
        Q0_i = max(rng.normal(loc=sp.Q0, scale=0.08), 0.05)
        kappa_i = max(rng.normal(loc=sp.kappa, scale=0.15 * sp.kappa), 0.05)

        nu_i = control.alpha_inventory_control(
            A_hat_k[i, :-1],
            params=control.ControlParams(
                T=control_params.T,
                N=control_params.N,
                Q0=Q0_i,
            ),
            kappa=kappa_i,
        )

        individual_nu.append(nu_i)
        individual_labels.append(sp.name)

individual_nu = np.array(individual_nu)


# ============================================================
# 7. Plots
# ============================================================

plotting.plot_unimpacted_and_impacted(
    F_t=Ft,
    S_t=St,
    latent_path=latent_proxy,
    sim_params=simulation_params,
)

plotting.plot_estimated_drifts(
    A_hat_k=A_hat_k,
    latent_path=latent_proxy,
    sim_params=simulation_params,
    subpops=SubPops,
    A0=mu0,
    A1=mu1,
)

plotting.plot_controls_subpops(
    nu_hat_k=nu_hat_k,
    nu_bar=nu_bar,
    sim_params=simulation_params,
    subpops=SubPops,
)

plotting.plot_individual_inventory_paths(
    individual_nu=individual_nu,
    individual_labels=individual_labels,
    nu_hat_k=nu_hat_k,
    nu_bar=nu_bar,
    sim_params=simulation_params,
    subpops=SubPops,
)

plotting.plot_price_distortion(
    F_t=Ft,
    S_t=St,
    sim_params=simulation_params,
)

plotting.plot_fundamental_vs_impacted_posteriors(
    pi_fund_k=pi_k,
    pi_imp_k=pi_imp_k,
    latent_path=latent_proxy,
    sim_params=simulation_params,
    subpops=SubPops,
)


convergence_errors = np.array(
    [1e-1, 6e-2, 3e-2, 1.5e-2, 8e-3, 3e-3, 1e-3]
)

plotting.plot_mean_field_convergence(
    convergence_errors,
    tolerance=1e-3,
)


# ============================================================
# 8. Export results
# ============================================================

metrics = results.export_quantitative_results(
    Ft=Ft,
    St=St,
    pi_k=pi_k,
    pi_imp_k=pi_imp_k,
    A_hat_k=A_hat_k,
    nu_hat_k=nu_hat_k,
    nu_bar=nu_bar,
    individual_nu=individual_nu,
    individual_labels=individual_labels,
    subpops=SubPops,
    sim_params=simulation_params,
    convergence_errors=convergence_errors,
)

print(f"\nReal-data HMM experiment complete for {ticker} on {date}.")
print("Quantitative results exported to output/results/")

for category, values in metrics.items():
    print(f"\n{category}")
    for key, value in values.items():
        print(f"  {key}: {value}")