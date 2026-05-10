# PFP reference data — source attribution

The CSV and JSON files in this directory are **derived data**, copied verbatim from
the companion repository
[`hirtatsu/beta-Sn-DFT-PFP-MEAM`](https://github.com/hirtatsu/beta-Sn-DFT-PFP-MEAM)
to allow side-by-side comparison of the foundation MLIPs benchmarked here against
the Preferred Potential (PFP v8) results reported in the paper:

> H. Tatsumi, A. M. Ito, A. Takayama, H. Nishikawa.
> *Comparison of Elastic Constants and Surface Energies of β-Sn from Density Functional
> Theory, Universal Machine Learning Potential, and Empirical Potentials.*
> *Modelling and Simulation in Materials Science and Engineering* (2026, in review).

## Files

| file | source | content |
|---|---|---|
| `cij_table.csv` | `beta-sn-method-comparison/elastic/data/cij_mape_vs_experiment.csv` | Elastic tensor (Cᵢⱼ + bulk modulus) for DFT/PBE, four PFP modes, three MEAM, with experimental reference (Rayne & Chandrasekhar 1960) and MAPE |
| `surface_energies.csv` | `beta-sn-method-comparison/surface/data/surface_energies_5faces.csv` | γ for five low-index faces × eight methods, with MAE vs DFT/PBE |
| `bulk_pfp_pbe_d3.json` | `beta-sn-method-comparison/surface/data/pfp_matlantis/bulk_pfp_pbe_d3.json` | PFP/PBE+D3 relaxed bulk: a=5.846, c=3.173 Å |

## Bulk lattice parameters (paper Table 1)

| method | a (Å) | c (Å) | c/a |
|---|---:|---:|---:|
| Experiment (ICSD 40037) | 5.831 | 3.182 | 0.546 |
| DFT/PBE | 5.970 | 3.218 | 0.539 |
| PFP/PBE | 5.929 | 3.201 | 0.540 |
| PFP/PBE+D3 | 5.846 | 3.173 | 0.543 |

## Use in this repository

`scripts/make_comparison.py` loads these CSV/JSON files to build the six-method
comparison tables and the six-panel Wulff figure. The two PFP modes are treated
as fixed reference values, not re-run; they are the canonical published numbers.

## Compute environment used to produce the source data

- **PFP**: Preferred Potential v8 on the Matlantis platform (Preferred Networks),
  modes `PBE` and `PBE_PLUS_D3`, `pfp_api_client.pfp.calculators.ase_calculator.ASECalculator`,
  ASE `ExpCellFilter` + `LBFGS`, fmax 0.001 eV/Å (bulk), 0.015 eV/Å (slab).
- **DFT**: OpenMX 3.9.9 on Plasma Simulator (NIFS), PBE19 norm-conserving
  pseudopotential, Sn7.0-s2p2d3f1 basis, real-space cutoff 200 Ry, k-mesh
  Δk ≈ 0.15 rad/Å, SCF NormRD < 1×10⁻⁷, force convergence 3×10⁻⁴ Ha/Bohr (slab).

See the source repository's `README.md` and `PAPER_SUMMARY.md` for the full protocol.
