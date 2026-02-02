# CV Strategy Performance Analysis (Dual-Winner)

## Best CV Performance Summary

| Algorithm | Parameters | Pure Accuracy | Balanced/Efficient | Sim | Gain | Time (s/n) | Sim Improv | Refinement Impact |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SPG** | const-0.5-2000-10.0-100 | Smart Adaptive | Smart Adaptive | 0.123 | 0 | Same | 4.16s/n | 18 | 30 Wins |
| **SPG** | const-0.9-1000-5.0-20 | Smart Adaptive | Inverse CV | 0.079 | +0.002 | 1.9x Faster | 0.14s/n | 3 | 8 Wins |
| **SPG** | exp-0.5-2000-10.0-100 | Regular CV | Smart Adaptive | 0.704 | +0.004 | 2.0x Faster | 1.90s/n | 7 | 27 Wins |
| **SPG** | exp-0.9-1000-5.0-20 | Regular CV | Regular CV | 0.812 | 0 | Same | 0.97s/n | 15 | 38 Wins |
| **SPGpCDSS** | const-0.5-2000-10.0-100 | Inverse CV | Inverse CV | 0.107 | 0 | Same | 3.14s/n | 21 | 43 Wins |
| **SPGpCDSS** | const-0.9-1000-5.0-20 | Smart Adaptive | Inverse CV | 0.077 | +0.004 | 10.9x Faster | 0.39s/n | 3 | 31 Wins |
| **SPGpCDSS** | exp-0.5-2000-10.0-100 | Smart Adaptive | Smart Adaptive | 0.697 | 0 | Same | 5.04s/n | 5 | 23 Wins |
| **SPGpCDSS** | exp-0.9-1000-5.0-20 | Smart Adaptive | Smart Adaptive | 0.861 | 0 | Same | 2.77s/n | 11 | 45 Wins |
| **L0LearnPSI1** | const-0.5-2000-10.0-100 | Smart Adaptive | Smart Adaptive | 0.098 | 0 | Same | 5.74s/n | 0 | 0 Wins |
| **L0LearnPSI1** | const-0.9-1000-5.0-20 | Smart Adaptive | Smart Adaptive | 0.044 | 0 | Same | 0.20s/n | 0 | 0 Wins |
| **L0LearnPSI1** | exp-0.5-2000-10.0-100 | Inverse CV | Inverse CV | 0.816 | 0 | Same | 79.19s/n | 0 | 0 Wins |
| **L0LearnPSI1** | exp-0.9-1000-5.0-20 | Smart Adaptive | Smart Adaptive | 0.662 | 0 | Same | 1.31s/n | 0 | 0 Wins |


## Refinement Impact Analysis

The **Refinement Duel** compares starting from the previous $\lambda$ result (**Path-Start**) vs. starting from zero (**Zero-Start**).

### Observations:
1. **SPG & SPGpCDSS**: Show high Z-Wins in the `exp` correlation cases. Zero-start is frequently essential to escape local minima.
2. **L0Learn**: Extremely stable; refinement rarely avoids path-stalling as the R implementation is robust.

## CV Strategy Duel: Smart Adaptive vs Inverse

| Algorithm | Parameters | Best Sim Δ | Speedup | Avg Time (Ad vs Inv) |
| :--- | :--- | :---: | :---: | :---: |
| SPG | const-0.5-2000-10.0-100 | +0.010 (Ad) | 2.4x (Inv Faster) | 4.16s vs 1.72s |
| SPG | const-0.9-1000-5.0-20 | +0.002 (Ad) | 1.9x (Inv Faster) | 0.26s vs 0.14s |
| SPG | exp-0.5-2000-10.0-100 | +0.003 (Ad) | 2.7x (Ad Faster) | 1.90s vs 5.21s |
| SPG | exp-0.9-1000-5.0-20 | +0.004 (Inv) | 2.0x (Inv Faster) | 0.82s vs 0.41s |
| SPGpCDSS | const-0.5-2000-10.0-100 | +0.000 (Inv) | 9.1x (Inv Faster) | 28.72s vs 3.14s |
| SPGpCDSS | const-0.9-1000-5.0-20 | +0.004 (Ad) | 10.9x (Inv Faster) | 4.24s vs 0.39s |
| SPGpCDSS | exp-0.5-2000-10.0-100 | +0.001 (Ad) | 1.3x (Ad Faster) | 5.04s vs 6.79s |
| SPGpCDSS | exp-0.9-1000-5.0-20 | +0.036 (Ad) | 2.7x (Ad Faster) | 2.77s vs 7.53s |

