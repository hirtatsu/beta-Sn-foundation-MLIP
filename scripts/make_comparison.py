"""Final comparison: MACE-MPA-0, ORB v3, SevenNet-Omni vs DFT/PBE/exp/PFP
   — all with paper-matching Tatsumi-2026 protocol."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from ase import Atoms
from wulffpack import SingleCrystal

ROOT = Path(str(Path(__file__).resolve().parent.parent))
RES = ROOT / "results"
FIG = ROOT / "figures"

methods = ['MACE_MPA0', 'SevenNet_Omni', 'ORB_v3']
data = {m: json.load(open(RES / f"benchmark_{m}.json")) for m in methods}

# Reference values from prior paper (Tatsumi 2026)
DFT_PBE = {'lat': {'a': 5.930, 'c': 3.201}, 'C': {'C11': 89.7, 'C33': 91.8, 'C12': 17.4, 'C13': 31.5, 'C44': 17.9, 'C66': 17.6, 'B_Voigt': 46.1},
           'gamma': {'100': 492.2, '101': 505.6, '110': 526.9, '111': 550.0, '001': 550.5}, 'mape_exp': 25.2}
PFP_PBE_D3 = {'C': {'C11': 98.5, 'C33': 121.1, 'C12': 36.2, 'C13': 36.5, 'C44': 22.9, 'C66': 16.2}, 'mape_exp': 24.2}
PFP_PBE = {'C': {'C11': 114.0, 'C33': 104.6, 'C12': 41.5, 'C13': 41.2, 'C44': 29.5, 'C66': 31.9}, 'mape_exp': 30.4}
EXP = {'C11': 73.4, 'C33': 90.7, 'C12': 59.1, 'C13': 35.8, 'C44': 22.0, 'C66': 24.0}

print("=" * 100)
print("  β-Sn benchmark — Tatsumi-2026 protocol (LBFGS, ε=±0.5%, internal relax)")
print("=" * 100)

print("\n[Bulk lattice]")
print(f"{'method':>15s}  {'a (Å)':>7s}  {'c (Å)':>7s}  {'c/a':>6s}")
print(f"{'Experiment':>15s}  {5.831:>7.3f}  {3.182:>7.3f}  {3.182/5.831:>6.3f}")
print(f"{'DFT/PBE':>15s}  {DFT_PBE['lat']['a']:>7.3f}  {DFT_PBE['lat']['c']:>7.3f}  "
      f"{DFT_PBE['lat']['c']/DFT_PBE['lat']['a']:>6.3f}")
print(f"{'PFP/PBE':>15s}  {5.929:>7.3f}  {3.201:>7.3f}  {3.201/5.929:>6.3f}")
for m in methods:
    a = data[m]['bulk']['lattice']['a']; c = data[m]['bulk']['lattice']['c']
    print(f"{m:>15s}  {a:>7.3f}  {c:>7.3f}  {c/a:>6.3f}")

print("\n[Elastic constants (Tatsumi-2026 protocol, GPa)]")
labs = ['C11', 'C33', 'C12', 'C13', 'C44', 'C66']
print(f"{'method':>16s} " + ' '.join(f'{l:>7s}' for l in labs) + '   MAPE_exp')
print(f"{'Experiment':>16s} " + ' '.join(f'{EXP[l]:>7.1f}' for l in labs))
print(f"{'DFT/PBE':>16s} " + ' '.join(f'{DFT_PBE["C"][l]:>7.1f}' for l in labs)
      + f'    {DFT_PBE["mape_exp"]:>5.1f}%')
print(f"{'PFP/PBE+D3':>16s} " + ' '.join(f'{PFP_PBE_D3["C"][l]:>7.1f}' for l in labs)
      + f'    {PFP_PBE_D3["mape_exp"]:>5.1f}%')
print(f"{'PFP/PBE':>16s} " + ' '.join(f'{PFP_PBE["C"][l]:>7.1f}' for l in labs)
      + f'    {PFP_PBE["mape_exp"]:>5.1f}%')
for m in methods:
    ti = data[m]['elastic']['tetragonal_independent_GPa']
    mape = data[m]['elastic']['MAPE_vs_exp_pct']
    print(f"{m:>16s} " + ' '.join(f'{ti[l]:>7.1f}' for l in labs)
          + f'    {mape:>5.1f}%')

print("\n[Surface energies (mJ/m²)]")
DFT_g = DFT_PBE['gamma']
print(f"{'method':>16s} " + ' '.join(f'({k:>3s})' for k in ['100','101','110','111','001'])
      + f"  {'MAE':>5s}  {'MAPE':>5s}")
print(f"{'DFT/PBE':>16s} " + ' '.join(f'{DFT_g[k]:>7.1f}' for k in ['100','101','110','111','001']))
for m in methods:
    res = data[m]['surfaces']['results']
    g = [res[k]['gamma_min_mJ_m2'] for k in ['100','101','110','111','001']]
    mae = data[m]['surfaces']['MAE_vs_DFT_PBE_mJ_m2']
    mape = data[m]['surfaces']['MAPE_vs_DFT_PBE_pct']
    print(f"{m:>16s} " + ' '.join(f'{x:>7.1f}' for x in g) + f'  {mae:>5.1f}   {mape:>4.1f}%')

print("\n[γ ordering (low → high)]")
print(f"  Experiment / DFT/PBE: 100 < 101 < 110 < 111 < 001")
for m in methods:
    res = data[m]['surfaces']['results']
    gd = {k: res[k]['gamma_min_mJ_m2'] for k in ['100','101','110','111','001']}
    print(f"  {m:>16s}: " + ' < '.join(sorted(gd, key=gd.get)))

# Wulff figure: 3 panels (DFT, MACE, SevenNet)
A_REF, C_REF = 5.970, 3.218
BETA_SN = Atoms('Sn4',
    scaled_positions=[[0,0,0],[0.5,0.5,0.5],[0,0.5,0.25],[0.5,0,0.75]],
    cell=[[A_REF,0,0],[0,A_REF,0],[0,0,C_REF]], pbc=True)
WULFF_COLORS = {(1,0,0): '#4C72B0', (1,0,1): '#8172B3', (1,1,0): '#C44E52',
                (1,1,1): '#CCB974', (0,0,1): '#55A868'}

cases = [('DFT/PBE',
          {(1,0,0): DFT_g['100'], (1,0,1): DFT_g['101'],
           (1,1,0): DFT_g['110'], (1,1,1): DFT_g['111'], (0,0,1): DFT_g['001']})]
for m in methods:
    res = data[m]['surfaces']['results']
    cases.append((m,
        {(1,0,0): res['100']['gamma_min_mJ_m2'],
         (1,0,1): res['101']['gamma_min_mJ_m2'],
         (1,1,0): res['110']['gamma_min_mJ_m2'],
         (1,1,1): res['111']['gamma_min_mJ_m2'],
         (0,0,1): res['001']['gamma_min_mJ_m2']}))

fig = plt.figure(figsize=(14, 4))
n_panels = len(cases)
for i, (label, gam) in enumerate(cases, 1):
    ax = fig.add_subplot(1, n_panels, i, projection='3d')
    sc = SingleCrystal(surface_energies=gam, primitive_structure=BETA_SN, natoms=10000)
    sc.make_plot(ax, alpha=0.88, linewidth=0.5, colors=WULFF_COLORS)
    gam_txt = ', '.join(f'({k[0]}{k[1]}{k[2]}):{int(round(v))}' for k, v in gam.items())
    ax.set_title(f'{label}\n{gam_txt}', fontsize=8)
    ax.view_init(elev=22, azim=35)
    ax.set_box_aspect([1, 1, 1])

handles = [Patch(color=c, label=f'({k[0]}{k[1]}{k[2]})') for k, c in WULFF_COLORS.items()]
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=10,
           bbox_to_anchor=(0.5, 0.00))
plt.suptitle('β-Sn Wulff shapes: DFT/PBE vs foundation MLIPs (paper-matching protocol)',
             fontsize=11, y=1.00)
plt.tight_layout(rect=[0, 0.06, 1, 0.97])
for ext in ('png', 'pdf'):
    plt.savefig(FIG / f'wulff_compare.{ext}', dpi=200, bbox_inches='tight')
print(f"\nSaved: {FIG / 'wulff_compare.png'}")
plt.close()
