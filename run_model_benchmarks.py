import subprocess
import time
from pathlib import Path

notebooks = [
    r"Final Benchmarks/Fashion-MNIST/Custom_CNN.ipynb",

    r"Final Benchmarks/GTSRB/ResNet-18.ipynb",
    r"Final Benchmarks/GTSRB/AlexNet.ipynb",
    r"Final Benchmarks/GTSRB/GoogLeNet.ipynb",

    r"Final Benchmarks/EuroSAT/ResNet-18.ipynb",
    r"Final Benchmarks/EuroSAT/AlexNet.ipynb",
    r"Final Benchmarks/EuroSAT/GoogLeNet.ipynb",

    r"Final Benchmarks/Imagenette/ResNet-18.ipynb",
    r"Final Benchmarks/Imagenette/AlexNet.ipynb",
    r"Final Benchmarks/Imagenette/GoogLeNet.ipynb",

    r"Final Benchmarks/Fashion-MNIST/ResNet-18.ipynb",
    r"Final Benchmarks/Fashion-MNIST/AlexNet.ipynb",
    r"Final Benchmarks/Fashion-MNIST/GoogLeNet.ipynb",

    r"Final Benchmarks/AG-News/ResNet-18.ipynb",
    r"Final Benchmarks/AG-News/AlexNet.ipynb",
    r"Final Benchmarks/AG-News/GoogLeNet.ipynb",
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
