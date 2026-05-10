"""Six-method comparison: DFT/PBE, PFP/PBE, PFP/PBE+D3, MACE-MPA-0, ORB v3, SevenNet-Omni.

All values aligned to paper:
  H. Tatsumi, A. M. Ito, A. Takayama, H. Nishikawa.
  "Comparison of Elastic Constants and Surface Energies of β-Sn from Density
   Functional Theory, Universal Machine Learning Potential, and Empirical Potentials"
  Modelling and Simulation in Materials Science and Engineering (2026, in review).

PFP and DFT reference data: results/pfp_reference/{cij_table.csv,surface_energies.csv}
copied from the companion repository hirtatsu/beta-Sn-DFT-PFP-MEAM.

MAPEs are recomputed inline against paper experimental Cij (Rayne & Chandrasekhar 1960:
C11=72.3, C12=59.4, C13=35.8, C33=88.4, C44=22.0, C66=24.0 GPa). The benchmark JSON
files store an older MAPE that used a slightly different experimental reference;
this script supersedes it.
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)
from matplotlib.patches import Patch
from ase import Atoms
from wulffpack import SingleCrystal

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PFP_REF = RES / "pfp_reference"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---------- Paper canonical reference values ----------
EXP_C = {'C11': 72.3, 'C12': 59.4, 'C13': 35.8, 'C33': 88.4, 'C44': 22.0, 'C66': 24.0}  # GPa
EXP_B = 55.0                                                                            # GPa, Voigt avg
EXP_LAT = {'a': 5.831, 'c': 3.182}                                                      # Å, ICSD 40037
EXP_GAMMA_AVG = 670.0                                                                   # mJ/m², Tyson–Miller 1977

# DFT/PBE (paper Tables 1–3)
DFT = {
    'lat': {'a': 5.970, 'c': 3.218},
    'C':   {'C11': 89.7, 'C12': 17.4, 'C13': 31.5, 'C33': 91.8, 'C44': 17.9, 'C66': 17.6, 'B': 47.8},
    'gamma_mJ': {'100': 492.0, '101': 506.0, '110': 527.0, '111': 550.0, '001': 551.0},
}

LABELS = ['C11', 'C12', 'C13', 'C33', 'C44', 'C66']
FACES_KEY = ['100', '101', '110', '111', '001']
FACES_HKL = {'100': (1, 0, 0), '101': (1, 0, 1), '110': (1, 1, 0), '111': (1, 1, 1), '001': (0, 0, 1)}


def mape_vs_exp(C):
    return float(np.mean([abs(C[k] - EXP_C[k]) / EXP_C[k] for k in LABELS]) * 100)


def mae_vs_dft(g):
    dft = np.array([DFT['gamma_mJ'][f] for f in FACES_KEY])
    arr = np.array([g[f] for f in FACES_KEY])
    return float(np.mean(np.abs(arr - dft)))


def mape_gamma_vs_dft(g):
    dft = np.array([DFT['gamma_mJ'][f] for f in FACES_KEY])
    arr = np.array([g[f] for f in FACES_KEY])
    return float(np.mean(np.abs(arr - dft) / dft) * 100)


# ---------- Load PFP reference (CSVs copied from companion repo) ----------
def load_pfp_cij(name):
    with open(PFP_REF / 'cij_table.csv') as f:
        for row in csv.DictReader(f):
            if row['method'] == name:
                return {k: float(row[k]) for k in LABELS} | {'B': float(row['B_Voigt'])}
    raise KeyError(name)


def load_pfp_gamma(name):
    with open(PFP_REF / 'surface_energies.csv') as f:
        for row in csv.DictReader(f):
            if row['method'] == name:
                return {f: float(row[f]) * 1000.0 for f in FACES_KEY}  # J/m² → mJ/m²
    raise KeyError(name)


PFP_PBE = {
    'lat': {'a': 5.929, 'c': 3.201},
    'C':   load_pfp_cij('PFP/PBE'),
    'gamma_mJ': load_pfp_gamma('PFP/PBE'),
}
PFP_PBE_D3 = {
    'lat': {'a': 5.846, 'c': 3.173},
    'C':   load_pfp_cij('PFP/PBE+D3'),
    'gamma_mJ': load_pfp_gamma('PFP/PBE+D3'),
}

# ---------- Foundation MLIP results (this repository) ----------
mlip_methods = ['MACE_MPA0', 'ORB_v3', 'SevenNet_Omni']
mlip_data = {m: json.load(open(RES / f"benchmark_{m}.json")) for m in mlip_methods}

mlip = {}
for m in mlip_methods:
    d = mlip_data[m]
    bulk = d['bulk']['lattice']
    Cd = d['elastic']['tetragonal_independent_GPa']
    surf = d['surfaces']['results']
    mlip[m] = {
        'lat': {'a': bulk['a'], 'c': bulk['c']},
        'C':   {k: float(Cd[k]) for k in LABELS} | {'B': float(Cd.get('B_Voigt', 0.0))},
        'gamma_mJ': {k: float(surf[k]['gamma_min_mJ_m2']) for k in FACES_KEY},
    }

# ---------- Print summary tables ----------
DISPLAY = [
    ('Experiment',     None),
    ('DFT/PBE',        DFT),
    ('PFP/PBE',        PFP_PBE),
    ('PFP/PBE+D3',     PFP_PBE_D3),
    ('MACE-MPA-0',     mlip['MACE_MPA0']),
    ('ORB v3',         mlip['ORB_v3']),
    ('SevenNet-Omni',  mlip['SevenNet_Omni']),
]

print("=" * 100)
print("β-Sn benchmark — paper-matching protocol (six-method comparison)")
print("Reference: Tatsumi et al., MSMSE (2026, in review)")
print("=" * 100)

print("\n[Bulk lattice]")
print(f"{'method':>16s} | {'a (Å)':>7s} {'c (Å)':>7s} {'c/a':>6s}")
print(f"{'Experiment':>16s} | {EXP_LAT['a']:>7.3f} {EXP_LAT['c']:>7.3f} {EXP_LAT['c']/EXP_LAT['a']:>6.3f}")
for label, d in DISPLAY[1:]:
    a, c = d['lat']['a'], d['lat']['c']
    print(f"{label:>16s} | {a:>7.3f} {c:>7.3f} {c/a:>6.3f}")

print("\n[Elastic constants (GPa) and bulk modulus B]")
hdr = f"{'method':>16s} | " + ' '.join(f'{l:>6s}' for l in LABELS) + f"  {'B':>5s}  {'MAPE':>5s}"
print(hdr)
print(f"{'Experiment':>16s} | " + ' '.join(f'{EXP_C[l]:>6.1f}' for l in LABELS) + f"  {EXP_B:>5.1f}  {'—':>5s}")
for label, d in DISPLAY[1:]:
    Cd = d['C']
    row = f"{label:>16s} | " + ' '.join(f'{Cd[l]:>6.1f}' for l in LABELS)
    row += f"  {Cd.get('B', float('nan')):>5.1f}"
    row += f"  {mape_vs_exp(Cd):>4.1f}%"
    print(row)

def rhu(x):  # round-half-up integer (paper convention; avoids Python banker's rounding)
    return int(np.floor(x + 0.5))


print("\n[Surface energies (mJ/m²)]")
hdr = f"{'method':>16s} | " + ' '.join(f'{f:>5s}' for f in FACES_KEY) + f"  {'MAE':>5s} {'MAPE':>5s}"
print(hdr)
for label, d in DISPLAY[1:]:
    g = d['gamma_mJ']
    row = f"{label:>16s} | " + ' '.join(f'{rhu(g[f]):>5d}' for f in FACES_KEY)
    if label == 'DFT/PBE':
        row += f"  {'—':>5s} {'—':>5s}"
    else:
        row += f"  {mae_vs_dft(g):>5.0f} {mape_gamma_vs_dft(g):>4.1f}%"
    print(row)

print("\n[γ ordering (low → high)]")
for label, d in DISPLAY[1:]:
    g = d['gamma_mJ']
    order = sorted(FACES_KEY, key=lambda k: g[k])
    print(f"  {label:>16s}: " + ' < '.join(f"({k})" for k in order))

# ---------- Wulff figure (six-panel, common reference lattice) ----------
A_REF, C_REF = DFT['lat']['a'], DFT['lat']['c']  # paper DFT/PBE lattice (5.970/3.218)
BETA_SN = Atoms('Sn4',
                scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5],
                                  [0, 0.5, 0.25], [0.5, 0, 0.75]],
                cell=[[A_REF, 0, 0], [0, A_REF, 0], [0, 0, C_REF]], pbc=True)

WULFF_COLORS = {
    (1, 0, 0): '#4C72B0', (1, 0, 1): '#8172B3', (1, 1, 0): '#C44E52',
    (1, 1, 1): '#CCB974', (0, 0, 1): '#55A868',
}


def wulff_dict(g):
    return {FACES_HKL[f]: g[f] for f in FACES_KEY}


panels = [
    ('DFT/PBE',       DFT['gamma_mJ']),
    ('PFP/PBE',       PFP_PBE['gamma_mJ']),
    ('PFP/PBE+D3',    PFP_PBE_D3['gamma_mJ']),
    ('MACE-MPA-0',    mlip['MACE_MPA0']['gamma_mJ']),
    ('ORB v3',        mlip['ORB_v3']['gamma_mJ']),
    ('SevenNet-Omni', mlip['SevenNet_Omni']['gamma_mJ']),
]

cols, rows = 3, 2
fig = plt.figure(figsize=(4.2 * cols, 4.4 * rows))
for i, (label, g) in enumerate(panels, 1):
    ax = fig.add_subplot(rows, cols, i, projection='3d')
    sc = SingleCrystal(surface_energies=wulff_dict(g),
                       primitive_structure=BETA_SN, natoms=2000)
    sc.make_plot(ax, alpha=0.88, linewidth=0.5, colors=WULFF_COLORS)
    gam_txt = ', '.join(f'({k}):{rhu(g[k])}' for k in FACES_KEY)
    ax.set_title(f'{label}\n{gam_txt}', fontsize=8)
    ax.view_init(elev=22, azim=35)
    ax.set_box_aspect([1, 1, 1])

handles = [Patch(color=c, label=f'({k[0]}{k[1]}{k[2]})') for k, c in WULFF_COLORS.items()]
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=10,
           bbox_to_anchor=(0.5, -0.01))
plt.suptitle('β-Sn Wulff shapes — DFT/PBE, PFP (PBE, PBE+D3), and three foundation MLIPs   '
             f'(common reference lattice a={A_REF:.3f}, c={C_REF:.3f} Å)',
             fontsize=10.5, y=1.00)
plt.tight_layout(rect=[0, 0.04, 1, 0.97])

for ext, dpi in (('png', 200), ('pdf', 300)):
    plt.savefig(FIG / f'wulff_compare.{ext}', dpi=dpi, bbox_inches='tight')
print(f"\nSaved: {FIG / 'wulff_compare.png'} (+ .pdf)")
plt.close(fig)
