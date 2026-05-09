"""β-Sn benchmark v2: EXACT methodology from Tatsumi 2026 β-Sn methods paper.

Reference: elastic_betasn.py from companion repo beta-Sn-DFT-PFP-MEAM.

Key parameters:
  - LBFGS (not BFGS) optimizer
  - Bulk: ExpCellFilter, hydrostatic_strain=False, fmax_cell=0.001 eV/Å
  - Elastic: atoms-only relax at fixed cell, LBFGS, fmax_atoms=0.005, Δε=±0.5%
  - Engineering shear convention: F[i,j] = δ_ij + 0.5*ε_voigt[off-diag]
  - Tetragonal 4/mmm symmetrization:
      C11 = (C00+C11)/2, C12 = (C01+C10)/2,
      C13 = (C02+C12+C20+C21)/4, C33 = C22, C44 = (C33+C44)/2, C66 = C55
  - Surface: FrechetCellFilter mask=[T,T,F,F,F,T], fmax=0.015 eV/Å
"""
import json, time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import write
from ase.optimize import LBFGS
from ase.filters import ExpCellFilter, FrechetCellFilter
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import SlabGenerator

EV_PER_A3_TO_GPA = 160.21766208
EV_TO_J = 1.602176634e-19
ANG2_TO_M2 = 1e-20

DFT_PBE_GAMMA = {'100': 492.2, '101': 505.6, '110': 526.9, '111': 550.0, '001': 550.5}
EXP_ELASTIC = {'C11': 73.4, 'C33': 90.7, 'C12': 59.1, 'C13': 35.8, 'C44': 22.0, 'C66': 24.0}
FACES = [(1,0,0), (1,0,1), (1,1,0), (1,1,1), (0,0,1)]

DELTA = 0.005     # ±0.5% strain (matches paper)
FMAX_CELL = 0.001
FMAX_ATOMS = 0.005   # looser than bulk relax (matches paper)
FMAX_SLAB = 0.015


def build_beta_sn(a=5.831, c=3.182):
    """β-Sn (I4_1/amd, Z=4) Wyckoff 4a + 4b in conventional cell."""
    return Atoms('Sn4',
        scaled_positions=[[0, 0, 0],
                          [0.5, 0.5, 0.5],
                          [0, 0.5, 0.25],
                          [0.5, 0, 0.75]],
        cell=[[a, 0, 0], [0, a, 0], [0, 0, c]], pbc=True)


def apply_voigt_strain(atoms, eps):
    """Apply Voigt strain [e1..e6] (engineering shear convention)."""
    F = np.array([
        [1.0 + eps[0], 0.5*eps[5], 0.5*eps[4]],
        [0.5*eps[5], 1.0 + eps[1], 0.5*eps[3]],
        [0.5*eps[4], 0.5*eps[3], 1.0 + eps[2]],
    ])
    new = atoms.copy()
    new.set_cell(atoms.cell @ F.T, scale_atoms=True)
    return new


def bulk_relax(calc, out_dir, method_name):
    sn = build_beta_sn()
    sn.calc = calc

    flt = ExpCellFilter(sn, hydrostatic_strain=False)
    opt = LBFGS(flt, logfile=str(out_dir / f"bulk_relax_{method_name}.log"))
    t0 = time.time()
    conv = opt.run(fmax=FMAX_CELL, steps=300)
    dt = time.time() - t0

    a, b, c = sn.cell.lengths()
    e = sn.get_potential_energy()
    mu = e / 4
    f = abs(sn.get_forces()).max()
    s_residual = sn.get_stress(voigt=True) * EV_PER_A3_TO_GPA

    write(out_dir / f"beta_Sn_bulk_{method_name}.cif", sn)
    rec = {
        "method": method_name,
        "lattice": {"a": float(a), "b": float(b), "c": float(c), "c_over_a": float(c/a)},
        "E_total_eV": float(e), "mu_Sn_eV_per_atom": float(mu),
        "fmax_final": float(f), "n_steps": int(opt.nsteps),
        "converged": bool(conv),
        "residual_stress_GPa": s_residual.tolist(),
        "wall_seconds": float(dt),
    }
    print(f"  Bulk: a={a:.4f}, c={c:.4f}, c/a={c/a:.4f}, μ_Sn={mu:.4f} eV "
          f"(steps={opt.nsteps}, t={dt:.1f}s, |stress|_max={abs(s_residual).max():.3f} GPa)")
    return sn, mu, rec


def elastic_tensor(calc, sn0):
    """Tatsumi 2026 elastic tensor (LBFGS, fmax_atoms=0.005, ±0.5%)."""
    print(f"  Elastic: ε=±{DELTA*100:.1f}%, atoms-only LBFGS at fmax_atoms={FMAX_ATOMS}")
    C = np.zeros((6, 6))
    t0 = time.time()
    for j in range(6):
        for sign in (+1, -1):
            eps = np.zeros(6); eps[j] = sign * DELTA
            a_strained = apply_voigt_strain(sn0, eps)
            a_strained.calc = calc
            LBFGS(a_strained, logfile=None).run(fmax=FMAX_ATOMS, steps=100)
            s = a_strained.get_stress(voigt=True) * EV_PER_A3_TO_GPA
            if sign == +1:
                s_plus = s
            else:
                s_minus = s
        C[:, j] = (s_plus - s_minus) / (2.0 * DELTA)
    dt = time.time() - t0

    # Tetragonal 4/mmm symmetrization (matches Tatsumi 2026 protocol)
    C11 = 0.5 * (C[0, 0] + C[1, 1])
    C12 = 0.5 * (C[0, 1] + C[1, 0])
    C13 = 0.25 * (C[0, 2] + C[1, 2] + C[2, 0] + C[2, 1])
    C33 = C[2, 2]
    C44 = 0.5 * (C[3, 3] + C[4, 4])
    C66 = C[5, 5]
    B_voigt = (2 * C11 + C33 + 2 * C12 + 4 * C13) / 9.0

    ind = {'C11': float(C11), 'C33': float(C33), 'C12': float(C12),
           'C13': float(C13), 'C44': float(C44), 'C66': float(C66),
           'B_Voigt': float(B_voigt)}
    mape = np.mean([abs(ind[k] - EXP_ELASTIC[k])/EXP_ELASTIC[k]
                    for k in EXP_ELASTIC]) * 100

    print(f"    C11={C11:.1f} C33={C33:.1f} C12={C12:.1f} "
          f"C13={C13:.1f} C44={C44:.1f} C66={C66:.1f} B={B_voigt:.1f} GPa")
    print(f"    MAPE_vs_exp={mape:.1f}%, t={dt:.1f}s")

    return {
        "tetragonal_independent_GPa": ind,
        "MAPE_vs_exp_pct": float(mape),
        "C_full_GPa_unsymmetrized": C.tolist(),
        "method": "Tatsumi-2026: LBFGS, fmax_atoms=0.005, Δε=±0.5%, internal relax",
        "wall_seconds": float(dt),
    }


def surface_energies(calc, bulk, mu_Sn, out_dir, slab_in_dir, slab_out_dir):
    bulk_pmg = AseAtomsAdaptor.get_structure(bulk)
    MASK = [True, True, False, False, False, True]
    print(f"  Surfaces: 5 faces (μ_Sn={mu_Sn:.4f} eV/atom)")
    results = {}
    t_total = time.time()
    for hkl in FACES:
        tag = ''.join(str(h) for h in hkl)
        sg = SlabGenerator(bulk_pmg, hkl, min_slab_size=15.0, min_vacuum_size=15.0,
                           center_slab=True, primitive=True, max_normal_search=2)
        slabs = sg.get_slabs(symmetrize=False, ftol=0.1)
        face_recs = []
        for ti, slab in enumerate(slabs):
            atoms = AseAtomsAdaptor.get_atoms(slab)
            write(slab_in_dir / f"slab_{tag}_t{ti}.cif", atoms)
            atoms.calc = calc
            flt = FrechetCellFilter(atoms, mask=MASK)
            opt = LBFGS(flt, logfile=str(out_dir / f"slab_{tag}_t{ti}.log"))
            t0 = time.time()
            conv = opt.run(fmax=FMAX_SLAB, steps=400)
            dt = time.time() - t0
            a1, b1, _ = atoms.cell.lengths()
            ang = atoms.cell.angles()[2]
            A_new = a1 * b1 * np.sin(np.radians(ang))
            e1 = atoms.get_potential_energy()
            N = len(atoms)
            gamma = (e1 - N * mu_Sn) * EV_TO_J / (2 * A_new * ANG2_TO_M2) * 1000
            write(slab_out_dir / f"slab_{tag}_t{ti}.cif", atoms)
            face_recs.append({"termination": ti, "N": int(N),
                              "A_A2": float(A_new), "E_eV": float(e1),
                              "gamma_mJ_m2": float(gamma),
                              "n_steps": int(opt.nsteps), "converged": bool(conv),
                              "wall_seconds": float(dt)})
        face_recs.sort(key=lambda r: r["gamma_mJ_m2"])
        gmin = face_recs[0]["gamma_mJ_m2"]
        results[tag] = {"all_terminations": face_recs, "gamma_min_mJ_m2": float(gmin)}
        print(f"    ({tag}): {len(slabs)} term, γ_min = {gmin:.1f} mJ/m²")

    mae = np.mean([abs(results[t]["gamma_min_mJ_m2"] - DFT_PBE_GAMMA[t])
                   for t in DFT_PBE_GAMMA])
    mape = np.mean([abs(results[t]["gamma_min_mJ_m2"] - DFT_PBE_GAMMA[t])
                    / DFT_PBE_GAMMA[t] for t in DFT_PBE_GAMMA]) * 100
    print(f"  → MAE={mae:.1f} mJ/m², MAPE={mape:.1f}% vs DFT/PBE  (total {time.time()-t_total:.1f}s)")
    return {"results": results,
            "MAE_vs_DFT_PBE_mJ_m2": float(mae),
            "MAPE_vs_DFT_PBE_pct": float(mape),
            "method": "Tatsumi-2026: FrechetCellFilter mask=[T,T,F,F,F,T], LBFGS, fmax=0.015"}


def run_full_benchmark(calc, method_name, root_dir):
    out = Path(root_dir) / "results"; out.mkdir(exist_ok=True, parents=True)
    slab_in = out / f"slabs_input_{method_name}"; slab_in.mkdir(exist_ok=True)
    slab_out = out / f"slabs_relaxed_{method_name}"; slab_out.mkdir(exist_ok=True)
    print(f"\n{'='*60}\n  {method_name} β-Sn benchmark v2 (Tatsumi 2026 protocol)\n{'='*60}")
    sn0, mu_Sn, bulk_rec = bulk_relax(calc, out, method_name)
    elastic_rec = elastic_tensor(calc, sn0)
    surf_rec = surface_energies(calc, sn0, mu_Sn, out, slab_in, slab_out)
    full = {"method": method_name, "protocol": "Tatsumi-2026",
            "bulk": bulk_rec, "elastic": elastic_rec, "surfaces": surf_rec}
    (out / f"benchmark_{method_name}.json").write_text(json.dumps(full, indent=2))
    print(f"  Saved: {out / f'benchmark_{method_name}.json'}")
    return full
