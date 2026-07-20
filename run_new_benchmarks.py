import subprocess
import time
from pathlib import Path

notebooks = [
    r"Final Benchmarks/EuroSAT/Custom_CNN.ipynb",
    r"Final Benchmarks/Imagenette/Custom_CNN.ipynb",
    r"Final Benchmarks/Oxford-IIIT-Pet/Custom_CNN.ipynb",
    r"Final Benchmarks/AG-News/Custom_TextCNN.ipynb",
    # GTSRB last: its download mirror (sid.erda.dk) is slow/flaky
    r"Final Benchmarks/GTSRB/Custom_CNN.ipynb"
]


total_start = time.time()

for nb in notebooks:

    path = Path(nb)

    if not path.exists():
        print(f"Missing file: {nb}")
        continue


    print("\n" + "="*70)
    print(f"STARTING: {nb}")
    print("="*70)

    start = time.time()

    try:
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--execute",
                "--to",
                "notebook",
                "--inplace",
                "--ExecutePreprocessor.timeout=-1",
                str(path)
            ],
            check=True
        )

        elapsed = time.time() - start

        print(f"FINISHED: {nb}")
        print(f"Runtime: {elapsed/60:.2f} minutes")


    except subprocess.CalledProcessError:

        print(f"FAILED: {nb}")
        print("Skipping to next notebook...")


total_time = time.time() - total_start

print("\n============================")
print("ALL POSSIBLE RUNS COMPLETE")
print(f"TOTAL TIME: {total_time/3600:.2f} hours")
print("============================")
