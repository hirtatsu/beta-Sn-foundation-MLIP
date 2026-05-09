# Foundation MLIP benchmark for β-Sn

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Benchmark of three open-source universal foundation machine-learning interatomic
potentials (**MACE-MPA-0**, **ORB v3**, **SevenNet-Omni**) against DFT/PBE on
**β-Sn** (tetragonal, I4₁/amd), focusing on:

- bulk lattice constants (a, c, c/a)
- elastic constant tensor (C₁₁, C₃₃, C₁₂, C₁₃, C₄₄, C₆₆)
- surface energies γ for five low-index faces (100), (101), (110), (111), (001)
- equilibrium Wulff shapes
- crystal-symmetry preservation under small strain

The benchmark uses the **same protocol** as Tatsumi et al. (in review at MSMSE,
"Comparison of Elastic Constants and Surface Energies of β-Sn"), which compared
DFT/PBE, four PFP modes, and three MEAM potentials. Here we extend that
benchmark to three of the leading **open-source foundation MLIPs** released
through 2026.

## Citation

If you use this benchmark, please cite the source paper:

> H. Tatsumi, A. M. Ito, A. Takayama, H. Nishikawa.
> *Comparison of Elastic Constants and Surface Energies of β-Sn*.
> Submitted to *Modelling and Simulation in Materials Science and Engineering* (2026).

A Zenodo DOI will be added on first stable release.

## Related repository

The DFT/PBE, PFP, and MEAM reference data of the source paper are available at
[**`hirtatsu/beta-Sn-DFT-PFP-MEAM`**](https://github.com/hirtatsu/beta-Sn-DFT-PFP-MEAM).

## Hardware

All inference run on a single NVIDIA RTX A4000 (16 GB VRAM, Ampere, CC 8.6),
PyTorch 2.5/2.11 (+CUDA 12.4/13.0 wheels). Each MLIP completes the full
benchmark (bulk + 12 strain configurations + 7 slab terminations) in < 1 minute.

## Methodology (paper-matching)

### Bulk relaxation
- Start from experimental ICSD 40037 (a = 5.831, c = 3.182 Å)
- ASE `ExpCellFilter(hydrostatic_strain=False)` + `LBFGS`
- Force convergence: |F|_max < 0.001 eV/Å

### Elastic tensor
- Apply Voigt strain at ±0.5% in 6 modes (xx, yy, zz, yz, xz, xy)
- For each strain: relax atomic positions only at fixed cell, LBFGS, fmax = 0.005 eV/Å
- Central difference: Cᵢⱼ = (σᵢ(+ε) − σᵢ(−ε)) / (2 Δε)
- Tetragonal 4/mmm symmetrization

### Surface energies
- Generate slabs from relaxed bulk via pymatgen `SlabGenerator`
  (full termination enumeration, min_slab_size = 15 Å, min_vacuum_size = 15 Å)
- Relax with `FrechetCellFilter(mask=[T,T,F,F,F,T])` + `LBFGS`, fmax = 0.015 eV/Å
- γ = (E_slab − N · μ_Sn) / (2 · A)
- Report minimum-γ termination per face

### Symmetry-preservation diagnostic
- Apply small strain ε_xx = +0.005 to relaxed bulk
- Atomic relax (cell fixed) and check drift of Sn z-coordinate
  (special Wyckoff position 4b at z = 1/4)

See [`METHODOLOGY.md`](METHODOLOGY.md) for full equations and rationale.

## Models tested

| Model | Loader | Training data | Released |
|---|---|---|---|
| **MACE-MPA-0** (medium) | `mace_mp(model='medium-mpa-0')` | MPtrj + Alexandria (PBE52) | 2025 |
| **ORB v3** (orb-v3-conservative-inf-omat) | `orb_v3_conservative_inf_omat()` | OMat24 | 2025/04 |
| **SevenNet-Omni** (modal=omat24) | `SevenNetCalculator(model='7net-omni', modal='omat24')` | 15 open ab initio datasets (13 protocols), multi-task | 2025 |

All three use float64 precision and run on a single CUDA device.

## Results

### Bulk lattice (vs DFT/PBE: a = 5.930, c = 3.201 Å)

| Method | a (Å) | c (Å) | c/a |
|---|---:|---:|---:|
| Experiment | 5.831 | 3.182 | 0.546 |
| DFT/PBE | 5.930 | 3.201 | 0.540 |
| MACE-MPA-0 | 5.957 | 3.205 | 0.538 |
| SevenNet-Omni | 5.963 | 3.206 | 0.538 |
| ORB v3 | 5.916 | **3.257** | **0.551** |

→ MACE-MPA-0 and SevenNet-Omni reproduce DFT/PBE within +0.5 % in `a`; ORB v3
slightly overestimates `c` and the c/a ratio.

### Elastic constants (paper-matching protocol, GPa)

| Method | C₁₁ | C₃₃ | C₁₂ | C₁₃ | C₄₄ | C₆₆ | MAPE vs exp |
|---|---:|---:|---:|---:|---:|---:|---:|
| Experiment | 73.4 | 90.7 | 59.1 | 35.8 | 22.0 | 24.0 | — |
| DFT/PBE | 89.7 | 91.8 | 17.4 | 31.5 | 17.9 | 17.6 | 25.2 % |
| PFP/PBE+D3 (paper best) | 98.5 | 121.1 | 36.2 | 36.5 | 22.9 | 16.2 | 24.2 % |
| PFP/PBE | 114.0 | 104.6 | 41.5 | 41.2 | 29.5 | 31.9 | 30.4 % |
| **MACE-MPA-0** | 68.7 | 79.6 | 41.5 | 26.5 | **6.9** | 13.8 | 30.9 % |
| **SevenNet-Omni** | 67.1 | 77.7 | 40.2 | 24.8 | **6.7** | 12.8 | 33.6 % |
| **ORB v3** | **−87.6** | 65.0 | **+200.6** | 40.9 | **0.5** | 10.2 | **109 %** |

→ **MACE-MPA-0** and **SevenNet-Omni** give physically reasonable but consistently
under-stiff Cᵢⱼ — they share the foundation-MLIP "softening artifact" of
under-estimating shear elastic constants C₄₄, C₆₆ by ~70 % relative to experiment.
**ORB v3** exhibits anomalous behavior on β-Sn under strain (negative C₁₁); this
arises from the symmetry-handling property discussed in the diagnostic below
and is a system-specific limitation of the off-the-shelf model.

### Surface energies (mJ/m²)

| Method | (100) | (101) | (110) | (111) | (001) | MAE | MAPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| DFT/PBE | 492.2 | 505.6 | 526.9 | 550.0 | 550.5 | — | — |
| **SevenNet-Omni** | **357.7** | 365.3 | 467.6 | 424.4 | 375.4 | **127.0** | **24.2 %** ★ |
| MACE-MPA-0 | 360.2 | 345.2 | 460.6 | 410.0 | 353.5 | 139.1 | 26.5 % |
| ORB v3 | 245.7 | 231.4 | 343.7 | 273.2 | 249.5 | 256.3 | 48.8 % |

### γ ordering (low → high)

| Method | Ordering | (100) lowest? |
|---|---|:-:|
| DFT/PBE | 100 < 101 < 110 < 111 < 001 | ✓ |
| **SevenNet-Omni** | **100** < 101 < 001 < 111 < 110 | ✓ |
| MACE-MPA-0 | 101 < 001 < 100 < 111 < 110 | ✗ |
| ORB v3 | 101 < 100 < 001 < 111 < 110 | ✗ |

### Wulff shapes

![Wulff shapes — DFT/PBE vs foundation MLIPs](figures/wulff_compare.png)

### Symmetry preservation under ε_xx = +0.005

| Method | Δz (Sn 4b) after atomic relax | ΔE (drifted vs symmetric) | Verdict |
|---|---:|---:|---|
| MACE-MPA-0 | +0.0041 | −0.083 meV/atom | ✓ preserved |
| SevenNet-Omni | +0.0043 | −0.078 meV/atom | ✓ preserved |
| **ORB v3** | **+0.0195** | **−0.563 meV/atom** | ✗ **broken** |

→ **ORB v3 favours an off-symmetric local minimum for β-Sn under small
strain**, lower in energy by 0.56 meV/atom than the symmetric configuration
(5× the drift and 7× the energy lowering observed for MACE / SevenNet). The
off-symmetric drift then propagates into the elastic-tensor calculation and
explains the anomalous Cᵢⱼ values reported above. This is a system-specific
behavior of the off-the-shelf ORB v3 on β-Sn rather than a numerical artifact
of the protocol.

## Conclusions

1. **MACE-MPA-0** and **SevenNet-Omni** preserve I4₁/amd symmetry under small
   strain and give physically reasonable elastic and surface properties for
   β-Sn, although both share the foundation-MLIP "softening artifact" of
   under-estimating the shear constants C₄₄, C₆₆ by ~70 %.
2. **ORB v3** (orb-v3-conservative-inf-omat) drives atomic positions off the
   special Wyckoff sites of β-Sn under small strain, which propagates into
   the elastic tensor as anomalous values (negative C₁₁, MAPE > 100 %) when
   atomic relaxation is allowed. For β-Sn–specific elastic-property
   predictions, the off-the-shelf ORB v3 should be used with caution and
   fine-tuned on system-specific data before deployment.
3. **Surface-energy MAE** vs DFT/PBE is 127 mJ/m² (SevenNet-Omni), 139 mJ/m²
   (MACE-MPA-0), and 256 mJ/m² (ORB v3). SevenNet alone reproduces the DFT
   ordering of (100) as the lowest-γ face.
4. **For β-Sn–specific fine-tuning workflows** (e.g., distillation from
   OpenMX reference data), **MACE-MPA-0** is selected as the foundation
   model on the basis of: (a) physically-honest behavior under strain,
   (b) mature distillation/fine-tuning ecosystem (LoRA + frozen-transfer
   learning), (c) robust LAMMPS ML-IAP integration for downstream MD,
   (d) broad coverage of Sn-IMC chemistry relevant to follow-on studies.
5. The shared C₄₄ deficiency of all three foundation MLIPs is the primary
   motivation for system-specific fine-tuning.

## Repository layout

```
.
├── README.md                          ← this file
├── METHODOLOGY.md                     ← full protocol / equations
├── CITATION.cff                       ← machine-readable citation metadata
├── LICENSE                            ← MIT
├── env/
│   └── requirements.txt               ← pinned package versions
├── scripts/
│   ├── lib_benchmark.py               ← shared benchmark library
│   ├── run_mace.py                    ← MACE-MPA-0 driver
│   ├── run_orb.py                     ← ORB v3 driver
│   ├── run_sevennet.py                ← SevenNet-Omni driver
│   ├── make_comparison.py             ← summary tables + Wulff figure
│   └── symmetry_test.py               ← symmetry-preservation diagnostic
├── results/
│   ├── benchmark_MACE_MPA0.json       ← all numbers (master)
│   ├── benchmark_ORB_v3.json
│   ├── benchmark_SevenNet_Omni.json
│   ├── beta_Sn_bulk_*.cif             ← MLIP-relaxed bulk structures
│   ├── slabs_input_*/                 ← initial slab CIFs (from pymatgen)
│   ├── slabs_relaxed_*/               ← MLIP-relaxed slab CIFs
│   ├── bulk_relax_*.log               ← LBFGS log per method
│   └── symmetry_test.json             ← symmetry diagnostic data
└── figures/
    └── wulff_compare.{png,pdf}        ← 4-panel Wulff figure (DFT + 3 MLIPs)
```

## Reproducing

```bash
# Set up environment (Python 3.10, CUDA 12.4)
python3.10 -m venv mlip-env
source mlip-env/bin/activate

pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
pip install mace-torch==0.3.15 'e3nn==0.4.4'    # MACE
pip install orb-models==0.5.5                    # ORB v3 (pulls torch 2.11+CUDA13)
pip install sevenn==0.12.1 'e3nn>=0.5.0'         # SevenNet (upgrades e3nn)
pip install pymatgen wulffpack matplotlib

# Run benchmarks (each ~1 min on RTX A4000)
python scripts/run_mace.py
python scripts/run_orb.py
python scripts/run_sevennet.py
python scripts/make_comparison.py
python scripts/symmetry_test.py
```

⚠️ ORB v3 (orb-models 0.5.5) pulls a newer torch wheel (2.11 + CUDA 13) that
conflicts with the torch 2.5 baseline used by MACE. The dependency-resolver
warnings are harmless if installed in the order above, but separate venvs
are cleaner if you intend to use one model heavily. ORB also requires
`LD_LIBRARY_PATH` to include `nvidia/cu13/lib` from inside its venv:
```bash
export LD_LIBRARY_PATH=$(pwd)/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```

## License

Code: MIT.
Data (CIFs, JSON results): CC-BY-4.0.

## Contact

For questions: tatsumi.jwri@osaka-u.ac.jp
