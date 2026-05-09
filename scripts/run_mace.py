from pathlib import Path
"""MACE-MPA-0 with Tatsumi-2026 exact protocol."""
import sys, warnings
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')

import torch
print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

from mace.calculators import mace_mp
print("Loading MACE-MPA-0...")
calc = mace_mp(model='medium-mpa-0', device='cuda', default_dtype='float64')

from lib_benchmark import run_full_benchmark
run_full_benchmark(calc, 'MACE_MPA0', str(Path(__file__).resolve().parent.parent))
