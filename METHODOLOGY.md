# Detailed methodology

This document records the exact protocol used in this benchmark, traceable to
the procedure of Tatsumi *et al.* (submitted to MSMSE, 2026), so that future
foundation MLIPs can be tested under identical conditions.

## β-Sn structure

- Space group: I4₁/amd (no. 141), origin choice 2
- Conventional cell: 4-atom primitive (Wyckoff 4a + 4b)
  - Sn₁ (4a): (0, 0, 0)
  - Sn₂ (4a): (1/2, 1/2, 1/2)
  - Sn₃ (4b): (0, 1/2, 1/4)
  - Sn₄ (4b): (1/2, 0, 3/4)
- Initial lattice (ICSD 40037, room T): a = 5.831 Å, c = 3.182 Å

## Bulk relaxation

- Filter: ASE `ExpCellFilter(hydrostatic_strain=False)`
- Optimizer: `LBFGS`
- Force criterion: |F|_max < 0.001 eV/Å
- Max steps: 300
- Stress at convergence: |σ|_max ~ 10⁻³–10⁻⁶ GPa (model-dependent)

## Elastic constants (Tatsumi-2026 protocol)

For each Voigt strain mode j ∈ {xx, yy, zz, yz, xz, xy} and sign s ∈ {+, −}:

1. Apply strain ε = s · 0.005 (engineering shear convention)

   ```
   F = [[1+ε₁,  ½ε₆,  ½ε₅],
        [½ε₆,  1+ε₂,  ½ε₄],
        [½ε₅,  ½ε₄,  1+ε₃]]
   ```

2. Strain the cell with `set_cell(cell @ F.T, scale_atoms=True)`

3. Relax atomic positions only (cell fixed):
   - Optimizer: `LBFGS`
   - Force criterion: |F|_max < 0.005 eV/Å
   - Max steps: 100
   - **Important**: looser fmax than bulk relax to avoid over-relaxation drift
     into off-symmetric local minima for foundation models with noisier energy
     landscapes (see "Symmetry preservation" below).

4. Read stress σ via `atoms.get_stress(voigt=True)`

5. Convert to GPa: × 160.21766208

Compute Cᵢⱼ via central difference:

```
Cᵢⱼ = (σᵢ(+ε) − σᵢ(−ε)) / (2 · 0.005)    in GPa
```

Tetragonal 4/mmm symmetrization (matches Tatsumi 2026 protocol):

```
C₁₁ = ½ (C₀₀ + C₁₁)
C₁₂ = ½ (C₀₁ + C₁₀)
C₁₃ = ¼ (C₀₂ + C₁₂ + C₂₀ + C₂₁)
C₃₃ = C₂₂
C₄₄ = ½ (C₃₃ + C₄₄)
C₆₆ = C₅₅
B_Voigt = (2 C₁₁ + C₃₃ + 2 C₁₂ + 4 C₁₃) / 9
```

## Surface energies

For each Miller plane (hkl) ∈ {(100), (101), (110), (111), (001)}:

1. Generate symmetrically distinct slabs from the relaxed bulk:
   - `pymatgen.core.surface.SlabGenerator(min_slab_size=15.0,
     min_vacuum_size=15.0, center_slab=True, primitive=True,
     max_normal_search=2)`
   - `get_slabs(symmetrize=False, ftol=0.1)`

2. Convert to ASE `Atoms` via `pymatgen.io.ase.AseAtomsAdaptor`

3. Relax with in-plane cell + atomic positions:
   - Filter: `FrechetCellFilter(mask=[T, T, F, F, F, T])`
     (relax xx, yy, xy components of cell; freeze zz, yz, xz to preserve vacuum)
   - Optimizer: `LBFGS`
   - Force criterion: |F|_max < 0.015 eV/Å
   - Max steps: 400

4. Compute γ:

   ```
   γ = (E_slab − N · μ_Sn) · e_to_J / (2 · A · Å²_to_m²)    in J/m²
       (here reported in mJ/m² × 1000)
   ```

   where μ_Sn = E_bulk / 4 from the relaxed 4-atom bulk cell, A is the
   in-plane area of the relaxed slab, and N is the number of atoms in the
   slab. Factor 2 accounts for two equivalent surfaces in the symmetric slab.

5. Report the minimum γ over symmetrically distinct terminations as the
   face's surface energy.

## Symmetry-preservation diagnostic

To check whether the foundation MLIP respects β-Sn's I4₁/amd symmetry under a
small strain:

1. Take the relaxed bulk
2. Apply ε_xx = +0.005 (uniaxial along a)
3. Relax atomic positions only with LBFGS, fmax = 0.001
4. Read final z fractional coordinate of Sn at site 4b (target z = 0.25)
5. Compute drift Δz = z_final − 0.25
6. Compute energy difference ΔE = E(drifted) − E(symmetric at +ε)

A physically honest potential should give |Δz| ≲ 0.005 with |ΔE|/atom ≲ 0.1
meV (numerical noise level). Larger drifts indicate the model has learned
spurious off-symmetric local minima — typically a training artifact.

## Reference values

| Property | Source |
|---|---|
| Experimental Cᵢⱼ | room-temperature β-Sn (consolidated in Tatsumi et al. MSMSE 2026, in review) |
| DFT/PBE Cᵢⱼ, γ | Tatsumi et al. (MSMSE in review, 2026), OpenMX with PBE19 norm-conserving pseudopotentials |
| PFP/PBE, PFP/PBE+D3 Cᵢⱼ | Same paper, PreFerred Potential v8 on Matlantis |
| MEAM (Ravelo, Etesami, Ko) Cᵢⱼ | Same paper, LAMMPS with NIST IPR potentials |
| Reference Wulff geometry | Same paper |

The DFT/PBE, PFP, and MEAM raw data and analysis scripts of the source paper
are openly available at
[`hirtatsu/beta-Sn-DFT-PFP-MEAM`](https://github.com/hirtatsu/beta-Sn-DFT-PFP-MEAM).

## Models tested (this benchmark)

| Model | Loader | Date | Reference |
|---|---|---|---|
| MACE-MPA-0 medium | `mace.calculators.mace_mp(model='medium-mpa-0', default_dtype='float64')` | 2025–2026 | github.com/ACEsuit/mace-foundations |
| ORB v3 (orb-v3-conservative-inf-omat) | `orb_models.forcefield.pretrained.orb_v3_conservative_inf_omat(precision='float64')` | 2025/4 | github.com/orbital-materials/orb-models |
| SevenNet-Omni (i12) | `sevenn.calculator.SevenNetCalculator(model='7net-omni', modal='omat24')` | 2025–2026 | github.com/MDIL-SNU/SevenNet |

## Deviations from Tatsumi 2026 paper

Negligible:
- All bulk lattice / elastic / γ formulas identical
- Same slab generation parameters
- Same LBFGS + filter + tolerance choices

Differences (foundation MLIP-specific):
- Foundation MLIPs use float64 precision (matching DFT/PBE in spirit; some MLIPs
  default to float32 for speed which we override).
- `pymatgen` SlabGenerator's `primitive=True, max_normal_search=2` parameters
  used for consistency across models (matching SlabGenerator defaults of the
  PFP/MEAM benchmark in the source paper).
