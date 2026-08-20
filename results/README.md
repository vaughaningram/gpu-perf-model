# Results

This directory contains curated, reproducible CSV results and plots. Large raw
profiler captures are intentionally excluded from version control.

`naive_a100_80gb_baseline.csv` is the first controlled M0 size sweep. Its rows
identify the exact measurement-code commit and A100 environment used.

`m1_a100_80gb_pcie_roofline_predictions.csv` freezes the analytical M1
predictions before comparison with that M0 sweep.
