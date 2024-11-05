from numba import cuda


def has_cuda() -> bool:
    return cuda.is_available()
