## Scripts

- `merge_unity_objects.ipynb` — Links Unity object-geometry information to the eye-tracking dataset so that object dimensions can be used in later analyses.

- `Size_Exclusion_FOV_Filter.ipynb` — Computes retinal image area estimates and applies plausibility or field-of-view based exclusion criteria.

- `fasterslower_retinal_area_glmms.Rmd` — Fits mixed-effects models for estimated retinal image area at the saccade origin and target as a function of residual class (Fast vs Slow), amplitude bin, and their interaction. The script log-transforms retinal image area, computes model diagnostics and Type II Wald chi-square tests, derives estimated marginal means, back-transforms them to deg² for plotting, and tests Holm-adjusted Fast/Slow contrasts within amplitude bins.

