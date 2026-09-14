# O3 — outlet/bloc differential error in the conflict frame

Lead-run 2026-09-12 on the 461 human-overlap units and the released labels. Answers panel R2-2:
"show the result under a sensitivity range for outlet/bloc-specific precision."

## Bloc-specific conflict error (weighted, 461 units)

| bloc | sensitivity | specificity |
|---|---:|---:|
| left | 0.971 | 0.662 |
| right | 0.994 | 0.713 |
| far_right | 1.000 | 0.546 |
| centre | 0.982 | 0.780 |

Sensitivity is flat. The differential lives entirely in specificity (false positives).

## Measurement-error sensitivity

Rogan-Gladen adjustment applied per bloc, with right-bloc prevalence weighted by right and far-right counts:

| | rho | p | n |
|---|---:|---:|---:|
| conflict gradient, observed labels | +0.5223 | 7.40e-03 | 25 |
| conflict gradient, bloc-error-adjusted | +0.5585 | 3.71e-03 | 25 |

Adjusting for the measured bloc-differential error strengthens the gradient slightly. The
measured differential is therefore not what produces it.

## Tipping point — what WOULD break it

We cannot estimate per-outlet error: 461 human units across 25 outlets is ~18 per outlet. So we
ask instead how large an outlet-level differential would have to be. Let outlet specificity drift
linearly with standardized outlet ideology, delta = the total specificity swing across the axis:

| delta | rho | p |
|---:|---:|---:|
| 0.00 | +0.5223 | 7.40e-03 |
| 0.05 | +0.3877 | 5.55e-02 |
| 0.10 | +0.0446 | 8.32e-01 |
| 0.15 | -0.3015 | 1.43e-01 |
| 0.20 | -0.5577 | 3.77e-03 |

**A specificity swing of about 0.10 across the ideological axis nulls the conflict gradient.**
That is smaller than the spread we observe between blocs (far-right 0.546 vs centre 0.780). We
have no per-outlet estimates and cannot exclude it. The honest conclusion is that the conflict
framing result is conditional on outlet-level false-positive rates not tracking outlet ideology
at a magnitude of ~0.10, and that this is a real and unresolved fragility, in contrast to the
tone gradient, which survives every check applied to it.
