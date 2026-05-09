from pathlib import Path
"""ORB v3 with Tatsumi-2026 exact protocol.

Note: ORB v3 (orb-v3-conservative-inf-omat) is included for methodological
completeness. As shown by the symmetry-preservation diagnostic
(scripts/symmetry_test.py), ORB v3 breaks I4₁/amd symmetry under ε_xx = +0.005
strain (Δz ≈ 0.020, ΔE ≈ -0.56 meV/atom), so atomic relaxation in the elastic
protocol drifts off-symmetric and the resulting Cᵢⱼ values are quantitatively
unreliable. We retain the run for transparency.
"""
import sys, warnings
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')

import torch
print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
print("Loading ORB-v3 (conservative-inf-omat)...")
orbff = pretrained.orb_v3_conservative_inf_omat(device='cuda', precision='float64')
calc = ORBCalculator(orbff, device='cuda')

from lib_benchmark import run_full_benchmark
run_full_benchmark(calc, 'ORB_v3', str(Path(__file__).resolve().parent.parent))
