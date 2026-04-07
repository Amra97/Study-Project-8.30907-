## Scripts

- `amplitude_mad_saccade_grouping.ipynb` — Classifies saccades as Fast or Slow relative to the expected amplitude-duration relationship. It uses participant-specific amplitude-duration modeling and MAD-based residual thresholds to identify saccades that are faster or slower than predicted for their amplitude.

- `amp_binning.ipynb` — Creates the amplitude bins used for the Fast/Slow residual-based analyses.

- `fasterslower_context_glmms.Rmd` — Fits the mixed-effects models used to test contextual differences between Fast and Slow saccades across amplitude bins. This script contains the main statistical analyses for binary and continuous contextual measures, including omnibus tests and Holm-adjusted within-bin contrasts.


