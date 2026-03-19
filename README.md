# Objective:

The goal of this project is to replicate the simulated results of
“Algorithmic Trading in Competitive Markets with Mean Field Games”
by Philippe Casgrain and Sebastian Jaimungal, and to analyze the behavior of heterogeneous agents trading under latent market signals and endogenous price impact.

Following replication, the project explores extensions such as model misspecification, finite-population effects, and sensitivity to market impact.

# Replication Targets:
## Inventory Dynamics
 - Heterogeneous agents with different urgancy levels
 - Subpopulation mean inventory vs individual trajectories
## Price Decomposition
  - Impacted vs. unimpacted price
  - Influence of aggregate order flow
## Posterier Filtering of Latent State
  - Estimation of hidden market regimes
  - Differences in beliefs accross agent subpopulations

# Model Overview:
## Latent Market
  - Hidden Markov model underlying price dynamics
## Price Dynamics
  - Midprice Evolves from order flow and latent factors
  - Agent Trades directly impact prices
## Agent Behavior
  - Continuous Trading at controlled rate
  - Strategy depends on inventory signals, interaction with mean field
## Objective
  - Utility function balancing profits, risk, liquiation costs
## Nash Equilibrium
  - Nash Equilibrium is approximated through MFG Framework

# Repo Structure:
- src- Core Simulation
- docs- Mathematical documentation of paper/model
- notebooks- Experiments, debugging
- results- Figures and Outputs
