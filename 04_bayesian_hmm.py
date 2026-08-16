import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def main():
    print("Loading regime features for Bayesian Analysis...")
    df = pd.read_csv("nifty_regime_features.csv", index_col=0, parse_dates=True)

    # Fast & Stable MCMC Sampling
    data_slice = df.tail(300).copy()
    returns = data_slice['ret_1d'].values

    K = 5  # 5 Regimes
    T = len(returns)

    print(f"Building Bayesian HMM Model in PyMC for {T} trading days...")

    with pm.Model() as bayesian_hmm_model:
        # 1. Transition Matrix Priors (Dirichlet)
        alpha_mat = np.eye(K) * 8.0 + (1 - np.eye(K)) * 1.0
        P = pm.Dirichlet('P', a=alpha_mat, shape=(K, K))
        
        # 2. Initial State Distribution
        pi = pm.Dirichlet('pi', a=np.ones(K), shape=K)
        
        # 3. Regime-conditional Return Means and Volatilities
        mu = pm.Normal('mu', mu=0.0, sigma=0.02, shape=K)
        sigma = pm.HalfNormal('sigma', sigma=0.03, shape=K)
        
        # 4. Latent Markov State Sequence
        states = pm.Categorical('states', p=pi, shape=T)
        
        # 5. Likelihood
        obs = pm.Normal('obs', mu=mu[states], sigma=sigma[states], observed=returns)

    print("\nSampling Posterior via NUTS Sampler (with target_accept=0.95)...")
    with bayesian_hmm_model:
        idata = pm.sample(
            draws=500, 
            tune=300, 
            chains=1, 
            cores=1, 
            target_accept=0.95, # Divergences कम करने के लिए
            random_seed=42, 
            return_inferencedata=True,
            progressbar=True
        )

    print("\nMCMC Sampling Completed Successfully!")

    # Fix: Fetch full summary without explicitly indexing hardcoded HDI column names
    summary = az.summary(idata, var_names=['mu', 'sigma'])
    print("\n--- Bayesian MCMC Diagnostics Summary ---")
    print(summary)

    # Save MCMC Diagnostic Plot
    az.plot_trace(idata, var_names=['mu', 'sigma'])
    plt.tight_layout()
    plt.savefig("bayesian_mcmc_trace.png")
    print("\nDiagnostic trace plot saved successfully as 'bayesian_mcmc_trace.png'!")

if __name__ == '__main__':
    main()
