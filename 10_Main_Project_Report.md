# Bayesian Regime Detection Engine for Equity Direction Forecasting
**Author:** Quantitative Data Analyst Trainee  
**Entity:** Zetheta Algorithms Private Limited (CIN: U62012MH2023PTC410415)  
**Target Application:** Long-Only Indian Mutual Fund Schemes (Asset Management Companies)

---

## EXECUTIVE SUMMARY
This project builds a Bayesian Regime Detection Engine that classifies the Indian equity market into discrete states (Risk-On, Risk-Off, Transitional, Late-Cycle, Post-Shock) with calibrated uncertainty. Over multi-month horizons in long-only Indian equity funds, point-forecast price prediction collapses due to low signal-to-noise ratios and non-stationarity. Instead, this engine uses a documented ensemble combining Hidden Markov Models (HMM), Bayesian HMMs via PyMC, Regime-Switching Vector Autoregressions (RS-VAR), Bayesian Neural Networks (BNN with MC Dropout), and Conformal Prediction wrappers.

---

## PART 1: DOMAIN FUNDAMENTALS & THE DIRECTIONAL THESIS
- **Indian Ecosystem:** Indian AUM crossed ₹70 lakh crore with monthly SIP inflows exceeding ₹26,000 crore. DIIs serve as the dominant price-setting force.
- **Direction Over Price:** Point forecasts ("Nifty will reach 26,000") are unprovable in real time and structurally miscalibrated[cite: 1]. Regime classification yields directional probabilities with explicit uncertainty budgets suitable for Investment Committee audits[cite: 1].
- **Regime Taxonomy:**
  1. *Risk-On:* Nifty > 200-DMA, VIX < 14, strong FII/DII inflows[cite: 1].
  2. *Risk-Off:* Drawdown, VIX > 22, persistent FII outflows, INR depreciation[cite: 1].
  3. *Transitional:* Mixed breadth, VIX 14-22, conflicting macro signals[cite: 1].
  4. *Late-Cycle:* Stretched valuations, narrow breadth, sector rotation[cite: 1].
  5. *Post-Shock:* Mean-reversion after acute drawdown, volatility decay[cite: 1].

---

## PART 2: MATHEMATICAL & STATISTICAL FOUNDATIONS
- **Bayesian Framework:**
  $$P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}$$
- **HMM State Dynamics:** Latent state sequence $S_t \in \{1,\dots,K\}$ governed by transition matrix $P_{ij} = P(S_t=j | S_{t-1}=i)$[cite: 1].
- **Uncertainty Decomposition:**
  - *Epistemic Uncertainty:* Model disagreement quantified via stochastic forward passes (MC Dropout) or ensemble variance[cite: 1].
  - *Aleatoric Uncertainty:* Irreducible market noise captured via predictive entropy[cite: 1].

---

## PART 3: MODEL ARCHITECTURE & ENSEMBLE DESIGN
1. **Frequentist HMM:** Baseline Gaussian emissions via `hmmlearn`[cite: 1].
2. **Bayesian HMM:** PyMC implementation using No-U-Turn Sampler (NUTS) with Dirichlet prior concentration on self-transitions[cite: 1].
3. **Regime-Switching VAR:** Captures cross-asset joint dynamics between Nifty, India VIX, USD/INR, and flows[cite: 1].
4. **Bayesian Neural Network:** Multi-layer perceptron with permanent dropout ($p=0.3$) evaluated over 200 Monte Carlo forward passes[cite: 1].
5. **Ensembling Layer:** Stacking meta-learner combining member outputs into a calibrated output contract[cite: 1].

---

## PART 4: BACKTEST PERFORMANCE & SEBI REGULATORY ALIGNMENT
- **Backtest Results (2019–2024):** The regime-aware tactical overlay de-risks during elevated crash probabilities, significantly boosting the Sharpe Ratio compared to passive Nifty Buy-and-Hold while curtailing max drawdowns[cite: 1].
- **SEBI Compliance:** Direct mapping to SEBI Risk-O-Meter disclosures, liquidity stress testing, and Stewardship Code documentation[cite: 1].
