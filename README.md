# FinMC

A repository containing code for CUDA-parallelised Monte Carlo simulations of barrier option pricing.

The code simulates a down-and-out call for a European barrier option with no rebate price with daily updates over 365 days. The simulation is parallelised over 1 million threads to be averaged at completion time.

## Compiling and running

Clone the repo and `cd` into the FinMC folder.
```
git clone https://github.com/gmguarino/FinMC.git
cd FinMC
```

Compile and run
```
make
make run
```

Clean Up
```
make clean
```

To run with plotting
```
make
make plot
make cleanplot
```


## Barrier Options

Barrier options are an exotic derivative whose payoff depends both on the price of the underlying asset at maturitym, as well as if the price hits a predefined level (barrier).

These can be split in **up** or **down** options, depending on whether the barrier is place above or below the spot price. Another split for there options is **in** and **out**. **In** refers to the activation of the option after reaching barrier level and **out** to its termination. Together with the distinction between **put** and **call** (sell and buy) this gives 8 different possible combinations for this type of option.

To simplify things only we are consider a European style option (the option can be exercised at maturity only) for a *down-and-out call*. Meaning the barrier is lower than the spot price and if the price is reduced under the barrier before maturity the option becomes invalid. There is no rebate price set for this option, so an invalid option will have a 0 payoff.

## The Simulation


The price is simulated using Euler's method which is a simple technique that does not grant incredible accuracy but is relatively straightforward to code up. Basically for each simulation, the price is updated as:

<img src="https://render.githubusercontent.com/render/math?math=S^{t\+1} - S^{t} = \mu S^t \Delta t + \sigma S^t \Delta W_t">

ie The increase in price is given by the estimated yearly return (<img src="https://render.githubusercontent.com/render/math?math=\mu">) multiplied by the current price and the relative time step, plus a noise term. This second term combines the expected volatility per year (<img src="https://render.githubusercontent.com/render/math?math=\sigma">) multiplied by the current price and a term which consists of a normally distributed random deviation (<img src="https://render.githubusercontent.com/render/math?math=\Delta W_t">), which is samples at every timestep for a distribution with 0 mean and variance <img src="https://render.githubusercontent.com/render/math?math=\Delta t">.