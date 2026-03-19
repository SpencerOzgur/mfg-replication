#Objective

The goal of this project is to replicate the simulated results of
“Algorithmic Trading in Competitive Markets with Mean Field Games”
by Philippe Casgrain and Sebastian Jaimungal, and to analyze the behavior of heterogeneous agents trading under latent market signals and endogenous price impact.

Following replication, the project explores extensions such as model misspecification, finite-population effects, and sensitivity to market impact.

Replication Targets:
- Inventory Dynamics
- Price Decomposition
- Posterier Filtering of Latent State

Model Overview:
- Accounts for Order Flow and Latent Factors
- Agents trade continuously at rate
- Midprice represented by Stochastic Differential Equation
- Agents Trade to maximize their own objective
- Goal is to obtain Nash Equilibrium

Repo Structure:
- src- reproducible source code
- docs- mathematical documentation of paper/model
