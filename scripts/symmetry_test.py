from pathlib import Path
"""Symmetry-preservation diagnostic: foundation MLIPs under small ε_xx strain.

Outputs Δz drift and energy lowering for each MLIP. See METHODOLOGY.md.
"""
import sys, json, warnings
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')

import numpy as np
from ase.io import read
from ase.optimize import LBFGS, BFGS

ROOT = str(Path(__file__).resolve().parent.parent)

def diagnose(calc, name, bulk_cif, optimizer=BFGS, fmax=0.001):
    sn = read(bulk_cif); sn.calc = calc
    a, c = sn.cell.lengths()[0], sn.cell.lengths()[2]
    z_init = sn.get_scaled_positions()[2, 2]
    e_bulk = sn.get_potential_energy()

    # +ε_xx strain
    sn_sym = sn.copy(); sn_sym.calc = calc
    F = np.eye(3); F[0, 0] = 1.005
    sn_sym.set_cell(sn.cell.array @ F.T, scale_atoms=True)
    e_sym = sn_sym.get_potential_energy()

    # Atomic relax
    sn_relax = sn_sym.copy(); sn_relax.calc = calc
    opt = optimizer(sn_relax, logfile=None)
    opt.run(fmax=fmax, steps=200)
    z_final = sn_relax.get_scaled_positions()[2, 2]
    e_relax = sn_relax.get_potential_energy()

    delta_z = z_final - 0.25
    delta_e_per_atom_meV = (e_relax - e_sym) / 4 * 1000

    rec = {
        "method": name,
        "bulk_lattice": {"a": float(a), "c": float(c)},
        "z_init": float(z_init),
        "z_after_relax_eps_xx_005": float(z_final),
        "delta_z": float(delta_z),
        "energy_drift_meV_per_atom": float(delta_e_per_atom_meV),
        "n_steps_relax": int(opt.nsteps),
        "verdict": "preserved" if abs(delta_z) < 0.005 else "BROKEN",
    }
    print(f"\n  {name}:")
    print(f"    bulk a={a:.4f}, c={c:.4f}")
    print(f"    z init = {z_init:.6f}")
    print(f"    after ε_xx=0.005 relax (steps={opt.nsteps}): z = {z_final:.6f}")
    print(f"    Δz = {delta_z:+.6f}")
    print(f"    ΔE = {delta_e_per_atom_meV:+.3f} meV/atom")
    print(f"    → {'✓ preserved' if abs(delta_z) < 0.005 else '✗ SYMMETRY BROKEN'}")
    return rec


def main():
    out = []

    # MACE-MPA-0
    try:
        from mace.calculators import mace_mp
        calc = mace_mp(model='medium-mpa-0', device='cuda', default_dtype='float64')
        out.append(diagnose(calc, 'MACE-MPA-0', f'{ROOT}/results/beta_Sn_bulk_MACE_MPA0.cif'))
    except Exception as e:
        print(f"  MACE failed: {e}")

    # ORB v3
    try:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        orbff = pretrained.orb_v3_conservative_inf_omat(device='cuda', precision='float64')
        calc = ORBCalculator(orbff, device='cuda')
        out.append(diagnose(calc, 'ORB v3', f'{ROOT}/results/beta_Sn_bulk_ORB_v3.cif'))
    except Exception as e:
        print(f"  ORB failed: {e}")

    # SevenNet-Omni
    try:
        from sevenn.calculator import SevenNetCalculator
        calc = SevenNetCalculator(model='7net-omni', modal='omat24', device='cuda')
        out.append(diagnose(calc, 'SevenNet-Omni', f'{ROOT}/results/beta_Sn_bulk_SevenNet_Omni.cif'))
    except Exception as e:
        print(f"  SevenNet failed: {e}")

    summary = {"protocol": "ε_xx = +0.005 strain, atomic relax with cell fixed (LBFGS, fmax=0.001)",
               "results": out}
    with open(f'{ROOT}/results/symmetry_test.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {ROOT}/results/symmetry_test.json")


if __name__ == '__main__':
    main()
