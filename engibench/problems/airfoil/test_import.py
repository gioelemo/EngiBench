# test_import.py
import sys

try:
    print("Attempting to import the slurm module...")
    import engibench.utils.slurm as slurm
    print("Import successful!")
    print(f"Available attributes: {dir(slurm)}")
except Exception as e:
    print(f"Import failed with error: {e}")
    import traceback
    traceback.print_exc()

# Try importing specific functions
try:
    from engibench.utils.slurm import SlurmConfig
    print("SlurmConfig import: SUCCESS")
except Exception as e:
    print(f"SlurmConfig import: FAILED - {e}")

try:
    from engibench.utils.slurm import sbatch_map
    print("sbatch_map import: SUCCESS")
except Exception as e:
    print(f"sbatch_map import: FAILED - {e}")