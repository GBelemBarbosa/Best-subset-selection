# CV Strategy Performance Analysis (Dual-Winner)

## Best CV Performance Summary

| Algorithm | Parameters | Pure Accuracy | Balanced/Efficient | Sim | Time (s) | Z-Wins | Z-Improv |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **SPG** | const-0.5-2000-10.0-100 | Smart Adaptive | Smart Adaptive | 0.218 | 6.7s | 2 | 2 |
| **SPG** | const-0.9-1000-5.0-20 | Inverse CV | Inverse CV | 0.174 | 0.1s | 0 | 0 |
| **SPG** | exp-0.5-2000-10.0-100 | Regular CV | Smart Adaptive | 1.000 | 5.7s | 5 | 0 |
| **SPG** | exp-0.9-1000-5.0-20 | Smart Adaptive | Inverse CV | 1.000 | 0.7s | 5 | 4 |
| **SPGpCDSS** | const-0.5-2000-10.0-100 | Inverse CV | Inverse CV | 0.172 | 4.1s | 5 | 3 |
| **SPGpCDSS** | const-0.9-1000-5.0-20 | Inverse CV | Inverse CV | 0.129 | 0.5s | 1 | 0 |
| **SPGpCDSS** | exp-0.5-2000-10.0-100 | Regular CV | Regular CV | 1.000 | 3.8s | 4 | 0 |
| **SPGpCDSS** | exp-0.9-1000-5.0-20 | Regular CV | Smart Adaptive | 1.000 | 6.8s | 3 | 0 |
| **L0LearnPSI1** | const-0.5-2000-10.0-100 | Regular CV | Regular CV | 0.190 | 70.7s | 0 | 0 |
| **L0LearnPSI1** | const-0.9-1000-5.0-20 | Smart Adaptive | Smart Adaptive | 0.116 | 0.9s | 0 | 0 |
| **L0LearnPSI1** | exp-0.5-2000-10.0-100 | Regular CV | Smart Adaptive | 1.000 | 2.6s | 0 | 0 |
| **L0LearnPSI1** | exp-0.9-1000-5.0-20 | Smart Adaptive | Smart Adaptive | 0.907 | 2.4s | 0 | 0 |


## Refinement Impact Analysis

The **Refinement Duel** compares starting from the previous $\lambda$ result (**Path-Start**) vs. starting from zero (**Zero-Start**).

### Observations:
1. **SPG & SPGpCDSS**: Show high Z-Wins in the `exp` correlation cases. Zero-start is frequently essential to escape local minima.
2. **L0Learn**: Extremely stable; refinement rarely avoids path-stalling as the R implementation is robust.
