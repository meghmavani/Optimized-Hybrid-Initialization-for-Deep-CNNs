import subprocess
import time
from pathlib import Path

notebooks = [
#    r"Final Benchmarks/CIFAR-10/CaffeNet.ipynb",
#    r"Final Benchmarks/CIFAR-10/Custom_CNN.ipynb",
#    r"Final Benchmarks/CIFAR-10/GoogLeNet-Small.ipynb",
#    r"Final Benchmarks/CIFAR-10/Mini-ResNet.ipynb",
#    r"Final Benchmarks/CIFAR-10/Mini-VGG.ipynb",
#    r"Final Benchmarks/CIFAR-10/Modified_VGG.ipynb",

#    r"Final Benchmarks/CIFAR-100/Mini-VGG.ipynb",
#    r"Final Benchmarks/CIFAR-100/ResNet-Mini.ipynb",
#    r"Final Benchmarks/CIFAR-100/VGG.ipynb",

#    r"Final Benchmarks/QMNIST/CaffeNet.ipynb",
#    r"Final Benchmarks/QMNIST/Custom_CNN.ipynb",
#    r"Final Benchmarks/QMNIST/GoogLeNet.ipynb",
#    r"Final Benchmarks/QMNIST/ResNet-18.ipynb",

#    r"Final Benchmarks/STL10/CaffeNet.ipynb",
#    r"Final Benchmarks/STL10/Modified_ResNet-18.ipynb",
#    r"Final Benchmarks/STL10/VGG.ipynb",

#    r"Final Benchmarks/SVHN/Mini-VGG.ipynb",
#    r"Final Benchmarks/SVHN/ResNet-Mini.ipynb",
#    r"Final Benchmarks/SVHN/VGG.ipynb"
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