# FinMC

A repository containing code for CUDA-parallelised Monte Carlo simulations of barrier option pricing.

The code simulates a down-and-out call for a European barrier option with no rebate price with daily updates over 365 days. The simulation is parallelised over 1 million threads to be averaged at completion time.