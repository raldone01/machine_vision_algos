# Machine Vision Algos

The machine vision alogs repository contains implementations for the following algorithms:

* Canny-End-To-End: Edge detector

## Setup

```shell
cd <project directory>
conda env create --file python/conda_env.yml --prefix .conda
```

## Canny-End-To-End

[`python/canny/canny_playground.ipynb`](python/canny/canny_playground.ipynb) is an interactive notebook that allows one to play around with the canny algorithm.

[`python/canny/canny_impls`](python/canny/canny_impls) contains multiple implementations that can be selected in the notebook.

### Numba Cuda FP32

This implementation runs on cuda using numba.
It runs on nvidia gpus.
The playground notebook has boolean toggle at the top to enable a cuda simulator.
It runs very slowly but works!

Find it here [`rd_numba_cuda_fp32.py`](python/canny/canny_impls/rd_numba_cuda_fp32.py).

### Vec v4 dibit

This implementation uses only numpy and scipy.
It runs on the cpu.

Find it here [`rd_vec_v4_dibit.py`](python/canny/canny_impls/rd_vec_v4_dibit.py).

### TODO:

* Canny-end-to-end
* Benchmark notebook
