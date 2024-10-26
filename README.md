# Machine Vision Algos

The machine vision algorithms repository contains implementations for the following algorithms:

* Canny-End-To-End: Edge detector

## Setup

```shell
cd <project directory>
conda env create --file python/conda_env.yml --prefix .conda
```

## Canny-End-To-End

[See more `python/canny`.](python/canny)

[`python/canny/canny_playground.ipynb`](python/canny/canny_playground.ipynb) is an interactive notebook that allows one to play around with the canny algorithm.

[`python/canny/canny_impls`](python/canny/canny_impls) contains multiple implementations that can be selected in the notebook.

### Numba Cuda FP32

This implementation runs on cuda using numba.
It runs on nvidia gpus.

The playground notebook has toggle at the top to enable a cuda simulator.
The simulator runs very slowly but works!

Find it here [`rd_numba_cuda_fp32.py`](python/canny/canny_impls/rd_numba_cuda_fp32.py).

### Vec v4 dibit

This is a vectorized canny implementation.
It uses only numpy and scipy.
It runs on the cpu.

Find it here [`rd_vec_v4_dibit.py`](python/canny/canny_impls/rd_vec_v4_dibit.py).

### TODO Implementations

* enable multiprocessing for opencv and numba on cpus
* numba stencil
* numba normal jit with loops
* opencv2 only

## TODO

* Benchmark notebook: Add system information like gpu, cpu, ram, ...

## Contributions

Contributions are welcome!
Open an issue before working on something big or complex.

## Usage as teaching material

Feel free to use anything in this repository as teaching materials.
If you do, I would love to be mentioned as a source.
An email informing me would also be appreciated.

Thank you.
