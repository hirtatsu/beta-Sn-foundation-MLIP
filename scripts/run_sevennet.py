from pathlib import Path
"""SevenNet-Omni with Tatsumi-2026 exact protocol."""
import sys, warnings
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')

import torch
print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

from sevenn.calculator import SevenNetCalculator
print("Loading SevenNet-Omni...")
calc = SevenNetCalculator(model='7net-omni', modal='omat24', device='cuda')

from lib_benchmark import run_full_benchmark
run_full_benchmark(calc, 'SevenNet_Omni', str(Path(__file__).resolve().parent.parent))
