"""
Canny end-to-end implementation using numba and cuda.
"""

# TODO: create a version with fp16 https://numba.readthedocs.io/en/stable/cuda-reference/kernel.html#bit-floating-point-intrinsics

import numpy as np
import numba as nb
from numba import cuda
import math
from utils.attr_dict import AttrDict
from numba.types import void, float32, uint8, uint32
from icecream import ic

TPB = 16


def is_cuda_array(obj):
    """Test if the object has defined the `__cuda_array_interface__` attribute.

    Does not verify the validity of the interface.
    """
    return hasattr(obj, "__cuda_array_interface__")


@cuda.jit(fastmath=True)
def _kernel_copy_devarray2d(input: np.array, output: np.array):
    x, y = cuda.grid(2)
    if x < input.shape[0] and y < input.shape[1]:
        output[x, y] = input[x, y]


@cuda.jit(fastmath=True, func_or_sig=void(uint8[:, :], float32[:, :]))
def _kernel_convert_to_float32(input_i: np.array, output_o: np.array):
    x, y = cuda.grid(2)
    if x < input_i.shape[0] and y < input_i.shape[1]:
        output_o[x, y] = input_i[x, y] / 255.0


@cuda.jit(
    fastmath=True, func_or_sig=void(float32[:, :], float32[:, :], float32), device=True
)
def _dev_gauss(image_i: np.array, image_o: np.array, sigma: float):
    x_thread, y_thread = cuda.threadIdx.x, cuda.threadIdx.y

    # compute the kernel radius
    one_dir = math.ceil(3.0 * sigma)
    kernel_width = int(2.0 * one_dir + 1.0)
    # TODO: The kernel should fit into a local array.
    # TODO: Check if it is faster if each thread computes the kernel on its own.
    # TODO: Check if its faster if the kernel is stored in constant memory or passed as an argument.
    # kernel = cuda.shared.array(shape=(kernel_width, kernel_width), dtype=np.float32)
    kernel = cuda.shared.array((100, 100), dtype=nb.types.float32)

    factor = 2.0 * math.pi * (sigma**2.0)
    # Loop over the kernel elements in chunks that fit in shared memory
    for i in range(x_thread, kernel_width, cuda.blockDim.x):
        for j in range(y_thread, kernel_width, cuda.blockDim.y):
            # Ensure we only compute values within bounds
            if i < kernel_width and j < kernel_width:
                x_k = i - one_dir
                y_k = j - one_dir
                kernel_val = (
                    math.exp(-(x_k**2.0 + y_k**2.0) / (2.0 * (sigma**2.0)))
                ) / factor
                kernel[i, j] = kernel_val

    # TODO: cache the block of the image in shared memory

    cuda.syncthreads()

    x, y = cuda.grid(2)

    # Perform convolution: Only execute if within image bounds
    x_width, y_height = image_i.shape
    if not (x < x_width and y < y_height):
        return

    one_dir = int(one_dir)

    result = 0.0
    # Now properly apply the kernel across relevant neighbors
    for i in range(-one_dir, one_dir + 1):
        for j in range(-one_dir, one_dir + 1):
            x_i = x + i
            y_j = y + j
            if x_i < 0:
                x_i = 0
            if x_i >= x_width:
                x_i = x_width - 1
            if y_j < 0:
                y_j = 0
            if y_j >= y_height:
                y_j = y_height - 1
            kernel_val = kernel[i + one_dir, j + one_dir] * image_i[x_i, y_j]
            result += kernel_val

    # Write back the result to the image
    # image_o[x, y] = result

    plot_kernel = False
    if plot_kernel:
        image_o[x, y] = result
        if x < kernel_width and y < kernel_width:
            image_o[x, y] = kernel[x, y] * factor
    else:
        image_o[x, y] = result


# TODO: check if its faster to use a two arrays instead of the io parameter
@cuda.jit(fastmath=True, func_or_sig=void(float32[:, :], float32[:, :], float32))
def _kernel_gauss(image_i: np.array, image_o: np.array, sigma: float):
    _dev_gauss(image_i, image_o, sigma)


def blur_gauss(image_u8_i: np.array, sigma: float) -> np.array:
    """Apply a Gaussian blur to the input image.

    :param image_u8_i: Input image in grayscale
    :type image_u8_i: np.array with shape (height, width) with dtype = np.uint8

    :param sigma: Standard deviation of the gaussian filter
    :type sigma: float

    :return: Blurred image
    :rtype: np.array with shape (height, width) with dtype = np.floating and values in the range [0., 1.]
    """

    # Disable warnings about low gpu utilization in the test suite
    old_cuda_low_occupancy_warnings = nb.config.CUDA_LOW_OCCUPANCY_WARNINGS
    nb.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

    stream = cuda.stream()

    height, width = image_u8_i.shape
    blockspergrid_x = math.ceil(height / TPB)
    blockspergrid_y = math.ceil(width / TPB)
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Check if the image buffer needs to be copied to the device
    input_on_device = is_cuda_array(image_u8_i)
    d_image_u8_i = (
        image_u8_i if input_on_device else cuda.to_device(image_u8_i, stream=stream)
    )

    # Convert input image to floating
    # Allocate a new array to store the floating image
    d_image_i = cuda.device_array(image_u8_i.shape, dtype=np.float32, stream=stream)
    _kernel_convert_to_float32[blockspergrid, (TPB, TPB), stream](
        d_image_u8_i, d_image_i
    )

    # Allocate output array
    d_blurred_o = cuda.device_array(image_u8_i.shape, dtype=np.float32, stream=stream)
    # Apply Gaussian filter
    _kernel_gauss[blockspergrid, (TPB, TPB), stream](d_image_i, d_blurred_o, sigma)

    blurred = (
        d_blurred_o if input_on_device else d_blurred_o.copy_to_host(stream=stream)
    )

    # Reset warnings
    nb.config.CUDA_LOW_OCCUPANCY_WARNINGS = old_cuda_low_occupancy_warnings

    stream.synchronize()

    return blurred


@cuda.jit(
    fastmath=True,
    func_or_sig=void(float32[:, :], float32[:, :], float32[:, :]),
    device=True,
)
def _dev_gradient_sobel(
    image_i: np.array, gradients_o: np.array, orientations_o: np.array
):
    # TODO: evaluate if the shared memory is faster (block cache)
    x, y = cuda.grid(2)

    x_width, y_height = image_i.shape

    if not (x < x_width and y < y_height):
        return

    # Compute the gradient

    # Assume sym boundary conditions

    # compute indices for the neighbours
    x_top_idx = x - 1
    if x_top_idx < 0:
        x_top_idx = 0
    y_left_idx = y - 1
    if y_left_idx < 0:
        y_left_idx = 0
    y_right_idx = y + 1
    if y_right_idx >= y_height:
        y_right_idx = y_height - 1
    x_bottom_idx = x + 1
    if x_bottom_idx >= x_width:
        x_bottom_idx = x_width - 1

    # compute the dx values for the neighbours
    dx = 0.0
    # top left
    dx -= image_i[x_top_idx, y_left_idx]
    # top right
    dx += image_i[x_top_idx, y_right_idx]
    # left
    dx += -2.0 * image_i[x, y_left_idx]
    # right
    dx += 2.0 * image_i[x, y_right_idx]
    # bottom left
    dx -= image_i[x_bottom_idx, y_left_idx]
    # bottom right
    dx += image_i[x_bottom_idx, y_right_idx]
    # compute the dy values for the neighbours
    dy = 0.0
    # top left
    dy += -image_i[x_top_idx, y_left_idx]
    # top
    dy += -2.0 * image_i[x_top_idx, y]
    # top right
    dy -= image_i[x_top_idx, y_right_idx]
    # bottom left
    dy += image_i[x_bottom_idx, y_left_idx]
    # bottom
    dy += 2.0 * image_i[x_bottom_idx, y]
    # bottom right
    dy += image_i[x_bottom_idx, y_right_idx]

    orientations_o[x, y] = np.atan2(dy, dx)
    gradient = math.sqrt(dx**2 + dy**2)
    # clip the gradient to the range [0, 1]
    # gradients_o[x, y] = np.clip(gradient, 0.0, 1.0) # scalars cause issues in numba
    gradients_o[x, y] = min(1.0, max(0.0, gradient))


@cuda.jit(fastmath=True, func_or_sig=void(float32[:, :], float32[:, :], float32[:, :]))
def _kernel_gradient_sobel(
    image_i: np.array, gradients_o: np.array, orientations_o: np.array
):
    _dev_gradient_sobel(image_i, gradients_o, orientations_o)


def sobel_gradients(image_i: np.array) -> tuple[np.array, np.array]:
    """Compute the gradients and orientations of the input image using the Sobel operator.

    :param image_i: Input image in grayscale this array will be overwritten with the gradients
    :type image_i: np.array with shape (height, width) with dtype = np.floating and values in the range [0., 1.]

    :return: (gradient, orientation): gradient: edge strength of the image in range [0.,1.],
                                      orientation: angle of gradient in range [-np.pi, np.pi]
    :rtype: (np.array, np.array)
    """

    # Disable warnings about low gpu utilization in the test suite
    old_cuda_low_occupancy_warnings = nb.config.CUDA_LOW_OCCUPANCY_WARNINGS
    nb.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

    stream = cuda.stream()

    height, width = image_i.shape

    blockspergrid_x = math.ceil(height / TPB)
    blockspergrid_y = math.ceil(width / TPB)
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Check if the image buffer needs to be copied to the device
    input_on_device = is_cuda_array(image_i)
    d_image_i = image_i if input_on_device else cuda.to_device(image_i, stream=stream)

    # Allocate output arrays
    d_gradients_o = cuda.device_array(image_i.shape, dtype=np.float32, stream=stream)
    d_orientations_o = cuda.device_array(image_i.shape, dtype=np.float32, stream=stream)

    _kernel_gradient_sobel[blockspergrid, (TPB, TPB), stream](
        d_image_i, d_gradients_o, d_orientations_o
    )

    gradients_o = (
        d_gradients_o if input_on_device else d_gradients_o.copy_to_host(stream=stream)
    )
    orientations_o = (
        d_orientations_o
        if input_on_device
        else d_orientations_o.copy_to_host(stream=stream)
    )

    # Reset warnings
    nb.config.CUDA_LOW_OCCUPANCY_WARNINGS = old_cuda_low_occupancy_warnings

    stream.synchronize()

    return gradients_o, orientations_o


@cuda.jit(
    fastmath=True,
    device=True,
    func_or_sig=void(float32[:, :], float32[:, :], float32[:, :]),
)
def _dev_non_max(gradients_i: np.array, orientations_i: np.array, edges_o: np.array):
    x, y = cuda.grid(2)
    x_width, y_height = gradients_i.shape

    if x >= x_width or y >= y_height:
        return

    gradient = gradients_i[x, y]
    orientation = orientations_i[x, y]

    # Compute neighbour mask: top, bottom sector
    mask_vertical = (
        ((orientation > 3 / 8 * np.pi) & (orientation <= 5 / 8 * np.pi))  # top
        | ((orientation > -5 / 8 * np.pi) & (orientation <= -3 / 8 * np.pi))  # bottom
    )
    gradient_right = 0
    if x + 1 < x_width:
        gradient_right = gradients_i[x + 1, y]
    gradient_left = 0
    if x - 1 >= 0:
        gradient_left = gradients_i[x - 1, y]
    mask = mask_vertical & (
        (gradient >= gradient_right)  # right
        & (gradient > gradient_left)  # left
    )

    # Compute neighbour mask: top right, bottom left sector
    mask_diag_1_tr_bl = (
        ((orientation > 1 / 8 * np.pi) & (orientation <= 3 / 8 * np.pi))  # top right
        | (
            (orientation > -7 / 8 * np.pi) & (orientation <= -5 / 8 * np.pi)
        )  # bottom left
    )
    gradient_top_left = 0
    if x - 1 >= 0 and y - 1 >= 0:
        gradient_top_left = gradients_i[x - 1, y - 1]
    gradient_bottom_right = 0
    if x + 1 < x_width and y + 1 < y_height:
        gradient_bottom_right = gradients_i[x + 1, y + 1]
    mask |= (
        mask_diag_1_tr_bl
        & (gradient > gradient_top_left)  # top left
        & (gradient > gradient_bottom_right)  # bottom right
    )

    # Compute neighbour mask: left, right sector
    mask_horizontal = (
        (orientation <= -7 / 8 * np.pi)  # left
        | (orientation > 7 / 8 * np.pi)  # left
        | ((orientation > -1 / 8 * np.pi) & (orientation <= 1 / 8 * np.pi))  # right
    )
    gradient_top = 0
    if y - 1 >= 0:
        gradient_top = gradients_i[x, y - 1]
    gradient_bottom = 0
    if y + 1 < y_height:
        gradient_bottom = gradients_i[x, y + 1]
    mask |= (
        mask_horizontal
        & (gradient > gradient_top)  # top
        & (gradient >= gradient_bottom)  # bottom
    )

    mask_diag_2_tl_br = (
        ((orientation > 5 / 8 * np.pi) & (orientation <= 7 / 8 * np.pi))  # top left
        | (
            (orientation > -3 / 8 * np.pi) & (orientation <= -1 / 8 * np.pi)
        )  # bottom right
    )
    gradient_top_right = 0
    if x + 1 < x_width and y - 1 >= 0:
        gradient_top_right = gradients_i[x + 1, y - 1]
    gradient_bottom_left = 0
    if x - 1 >= 0 and y + 1 < y_height:
        gradient_bottom_left = gradients_i[x - 1, y + 1]
    mask |= (
        mask_diag_2_tl_br
        & (gradient > gradient_top_right)  # top right
        & (gradient > gradient_bottom_left)  # bottom left
    )

    edges_o[x, y] = mask * gradient


@cuda.jit(fastmath=True, func_or_sig=void(float32[:, :], float32[:, :], float32[:, :]))
def _kernel_non_max(gradients_i: np.array, orientations_o: np.array, edges_o: np.array):
    _dev_non_max(gradients_i, orientations_o, edges_o)


def non_max(gradients_i: np.array, orientations_i: np.array) -> np.array:
    """Apply Non-Maxima Suppression and return an edge image.

    Filter out all the values of the gradients array which are not local maxima.
    The orientations are used to check for larger pixel values in the direction of orientation.

    :param gradients_i: Edge strength of the image in range [0.,1.] this array will be overwritten with the non-maxima suppressed gradients
    :type gradients_i: np.array

    :param orientations_i: angle of gradient in range [-np.pi, np.pi]
    :type orientations_i: np.array

    :return: Edge image with values between 0 and 1
    :rtype: np.array with shape (height, width) with dtype = np.floating
    """

    # Disable warnings about low gpu utilization in the test suite
    old_cuda_low_occupancy_warnings = nb.config.CUDA_LOW_OCCUPANCY_WARNINGS
    nb.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

    stream = cuda.stream()

    height, width = gradients_i.shape

    blockspergrid_x = math.ceil(height / TPB)
    blockspergrid_y = math.ceil(width / TPB)
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Check if the gradients and orientations are already on the device
    gradients_i_on_device = is_cuda_array(gradients_i)
    orientations_i_on_device = is_cuda_array(orientations_i)
    any_input_on_device = gradients_i_on_device or orientations_i_on_device

    d_gradients_i = (
        gradients_i
        if gradients_i_on_device
        else cuda.to_device(gradients_i, stream=stream)
    )

    d_orientations_i = (
        orientations_i
        if orientations_i_on_device
        else cuda.to_device(orientations_i, stream=stream)
    )

    # Allocate output arrays
    d_edges_o = cuda.device_array((height, width), dtype=np.float32, stream=stream)

    _kernel_non_max[blockspergrid, (TPB, TPB), stream](
        d_gradients_i, d_orientations_i, d_edges_o
    )

    edges_o = (
        d_edges_o if any_input_on_device else d_edges_o.copy_to_host(stream=stream)
    )

    # Reset warnings
    nb.config.CUDA_LOW_OCCUPANCY_WARNINGS = old_cuda_low_occupancy_warnings

    stream.synchronize()

    return edges_o


HISTOGRAM_BIN_COUNT = 256


# https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
@cuda.jit(fastmath=True, func_or_sig=void(float32[:, :], uint32[:]))
def _kernel_compute_edge_histogram_partial(
    edges_i: np.array, partial_histograms_o: np.array
):
    x_width, y_height = edges_i.shape

    # pixel coordinates
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # grid dimensions
    nx = cuda.gridDim.x * cuda.blockDim.x
    ny = cuda.gridDim.y * cuda.blockDim.y

    # linear thread index within 2D block
    linear_tid = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x

    # total number of threads in the 2D block
    BLOCK_THREADS = cuda.blockDim.x * cuda.blockDim.y

    # linear block index within 2D grid
    g = cuda.blockIdx.x + cuda.blockIdx.y * cuda.gridDim.x

    # Initialize shared memory
    shared = cuda.shared.array(shape=HISTOGRAM_BIN_COUNT, dtype=nb.types.uint32)
    # Initialize the histogram bins to zero
    for i in range(linear_tid, HISTOGRAM_BIN_COUNT, BLOCK_THREADS):
        shared[i] = 0
    cuda.syncthreads()

    # Process the edges
    # NOTE: There is a faster way to do this using radix sort and tracking the discontinuities but it is way more complex
    # https://github.com/NVIDIA/cccl/blob/a8dd6912d080173ff731c0e79a8a87647164ecd8/cub/cub/block/block_histogram.cuh#L296

    # Write our block's partial histogram to shared memory
    for col in range(x, x_width, nx):
        for row in range(y, y_height, ny):
            edge_int = uint32(edges_i[col, row] * HISTOGRAM_BIN_COUNT)
            edge_int = min(HISTOGRAM_BIN_COUNT - 1, edge_int)
            if edge_int == 0:
                # Skip the zero bin. That's where all the suppressed edges fall into.
                # That causes a lot of collisions and slows down the histogram computation.
                # And we don't need the zero bin anyway.
                continue
            cuda.atomic.add(shared, edge_int, 1)
    cuda.syncthreads()

    # Write the partial histogram into the global memory
    global_memory_histogram_offset = g * HISTOGRAM_BIN_COUNT
    for i in range(linear_tid, HISTOGRAM_BIN_COUNT, BLOCK_THREADS):
        partial_histograms_o[global_memory_histogram_offset + i] = shared[i]


@cuda.jit(fastmath=True, func_or_sig=void(uint32[:], float32[:], float32[:]))
def _kernel_compute_edge_histogram_final_accum(
    partial_histograms_i: np.array,
    low_high_prop_i: np.array,
    low_high_thresholds_o: np.array,
):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # Make sure the low and high thresholds are initialized
    if x < 2:
        low_high_thresholds_o[x] = 69.0

    x_partial_histogram_count = partial_histograms_i.shape[0] // HISTOGRAM_BIN_COUNT

    # if x == 0:
    #    print("x_partial_histogram_count", x_partial_histogram_count)

    # NOTE: These are essentially HISTOGRAM_BIN_COUNT parallel cumulative sums
    # https://people.cs.vt.edu/yongcao/teaching/cs5234/spring2013/slides/Lecture10.pdf
    # https://en.wikipedia.org/wiki/Prefix_sum
    # https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

    # Sum all the partial histograms
    histogram_non_cumulative = cuda.shared.array(
        shape=HISTOGRAM_BIN_COUNT, dtype=nb.types.uint32
    )
    if x < HISTOGRAM_BIN_COUNT:
        total = 0
        for i in range(x_partial_histogram_count):
            total += partial_histograms_i[x + i * HISTOGRAM_BIN_COUNT]
        histogram_non_cumulative[x] = total
        # print("X", x, "total", total)
    cuda.syncthreads()

    # Calculate the final cumulative histogram
    histogram_cumulative = cuda.shared.array(
        shape=HISTOGRAM_BIN_COUNT, dtype=nb.types.uint32
    )
    if x < HISTOGRAM_BIN_COUNT:
        total = 0
        # The first bucket is empty anyway
        for i in range(1, x + 1):
            total += histogram_non_cumulative[i]
        histogram_cumulative[x] = total
        # print("C", x, "total", total)

    cuda.syncthreads()

    low_prop, high_prop = low_high_prop_i[0], low_high_prop_i[1]

    total_pixels = histogram_cumulative[HISTOGRAM_BIN_COUNT - 1]
    low_pixels = total_pixels * (1.0 - low_prop)
    high_pixels = total_pixels * (1.0 - high_prop)

    if x == 0:
        print(
            "total_pixels",
            total_pixels,
            "low_pixels",
            low_pixels,
            "high_pixels",
            high_pixels,
        )

    # Every thread looks at his own bucket and the next one to decide if it is the low or high threshold
    if x < HISTOGRAM_BIN_COUNT:
        bucket = histogram_cumulative[x]
        if bucket <= low_pixels:
            # We might have found the low threshold.
            # Check if the next bucket is above the low threshold.
            next_bucket_idx = x + 1
            if next_bucket_idx < HISTOGRAM_BIN_COUNT:
                next_bucket = histogram_cumulative[next_bucket_idx]
            else:
                next_bucket = 0xFFFFFFFF
            print("LC", x, bucket, next_bucket)
            if next_bucket >= low_pixels:
                print("L", x, next_bucket)
                low_high_thresholds_o[0] = x  # / HISTOGRAM_BIN_COUNT
        if bucket <= high_pixels:
            # We might have found the high threshold.
            # Check if the next bucket is above the high threshold.
            next_bucket_idx = x + 1
            if next_bucket_idx < HISTOGRAM_BIN_COUNT:
                next_bucket = histogram_cumulative[next_bucket_idx]
            else:
                next_bucket = 0xFFFFFFFF
            print("HC", x, bucket, next_bucket)
            if next_bucket >= high_pixels:
                print("H", x, next_bucket)
                low_high_thresholds_o[1] = x  # / HISTOGRAM_BIN_COUNT


def compute_hysteresis_auto_thresholds(
    gradients_i: np.array, low_high_prop_i: np.array
) -> tuple[float, float]:
    """Compute the hysteresis thresholds based on the gradient strength.

    :param gradients_i: Edge strength of the image in range [0.,1.]
    :type gradients_i: np.array

    :param low_high_prop: Array with the proportion of the lowest and highest gradient values to be used as the low and high threshold
    :type low_high_prop: np.array with shape (2,) with dtype = np.float32

    :return: (low, high): [0]: Low threshold for the hysteresis, [1]: High threshold for the hysteresis
    :rtype: np.array with shape (2,) with dtype = np.floating
    """

    # Disable warnings about low gpu utilization in the test suite
    old_cuda_low_occupancy_warnings = nb.config.CUDA_LOW_OCCUPANCY_WARNINGS
    nb.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

    stream = cuda.stream()

    height, width = gradients_i.shape
    blockspergrid_x = math.ceil(height / TPB)
    blockspergrid_y = math.ceil(width / TPB)
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Check if the image buffer needs to be copied to the device
    input_on_device = is_cuda_array(gradients_i)
    d_gradients_i = (
        gradients_i if input_on_device else cuda.to_device(gradients_i, stream=stream)
    )

    # Check if the low_high_prop buffer needs to be copied to the device
    low_high_prop_on_device = is_cuda_array(low_high_prop_i)
    d_low_high_prop_i = (
        low_high_prop_i
        if low_high_prop_on_device
        else cuda.to_device(low_high_prop_i, stream=stream)
    )

    # Allocate output arrays
    d_low_high_thresholds_o = cuda.device_array(2, dtype=np.float32, stream=stream)
    histogram_count = blockspergrid_x * blockspergrid_y

    ic(histogram_count)

    d_partial_histograms = cuda.device_array(
        histogram_count * HISTOGRAM_BIN_COUNT,
        dtype=np.uint32,
        stream=stream,
    )

    # Compute the auto thresholds
    _kernel_compute_edge_histogram_partial[blockspergrid, (TPB, TPB), stream](
        d_gradients_i, d_partial_histograms
    )
    _kernel_compute_edge_histogram_final_accum[
        1, max(histogram_count, HISTOGRAM_BIN_COUNT), stream
    ](d_partial_histograms, d_low_high_prop_i, d_low_high_thresholds_o)

    low_high_thresholds_o = (
        d_low_high_thresholds_o
        if input_on_device
        else d_low_high_thresholds_o.copy_to_host(stream=stream)
    )

    # Reset warnings
    nb.config.CUDA_LOW_OCCUPANCY_WARNINGS = old_cuda_low_occupancy_warnings

    stream.synchronize()

    return low_high_thresholds_o


@cuda.jit(fastmath=True, device=True, func_or_sig=void(float32[:, :], float32, float32))
def _dev_hysteresis(
    gradients_io: np.array, low_threshold: np.array, high_threshold: np.array
):
    """
    Apply hysteresis thresholding to the gradients array.
    We do ab breadth-first search to find connected edges.
    All edges under the low threshold are discarded.
    We mark all edges above the high threshold as edges.
    Then we iterate and look at all neighbours of every pixel.
    If we are below the low threshold we discard the pixel.
    If we find a neighbour that is an edge we mark ourselves as an edge.
    If an iteration does not find any new edges we are done.
    """

    x, y = cuda.grid(2)
    x_width, y_height = gradients_io.shape

    if x >= x_width or y >= y_height:
        return

    if gradients_io[x, y] > high_threshold:
        # snap all pixels above the high threshold to 1.0 to mark them as edges
        gradients_io[x, y] = 1.0
        return

    if gradients_io[x, y] < low_threshold:
        # discard all pixels below the low threshold
        gradients_io[x, y] = 0.0
        return

    new_edges_found = cuda.shared.array(1, np.bool)
    new_edges_found[0] = False

    while not new_edges_found[0]:
        for i in range(-1, 2):
            for j in range(-1, 2):
                x_i = x + i
                y_j = y + j
                if x_i >= 0 and x_i < x_width and y_j >= 0 and y_j < y_height:
                    if gradients_io[x_i, y_j] == 1.0:
                        gradients_io[x, y] = 1.0
                        new_edges_found[0] = True
                        return
        cuda.syncthreads()

    # Discard all pixels that were not marked as edges
    if gradients_io[x, y] != 1.0:
        gradients_io[x, y] = 0.0


@cuda.jit(fastmath=True, func_or_sig=void(float32[:, :], float32, float32))
def _kernel_hysteresis(
    gradients_io: np.array, low_threshold: float, high_threshold: float
):
    _dev_hysteresis(gradients_io, low_threshold, high_threshold)


def canny_edge_detection(
    image_u8_i: np.array,
    sigma: float,
    low_high: np.array,
    auto_threshold: bool = False,
) -> np.array:
    """Apply Canny edge detection to the input image.

    :param image_u8_i: Input image in grayscale
    :type image_u8_i: np.array with shape (height, width) with dtype = np.uint8

    :param sigma: Standard deviation of the gaussian filter
    :type sigma: float

    :param low_high: Array with the low and high threshold for the hysteresis. If auto_threshold is True this array will be used as the proportion of the lowest and highest gradient values to be used as the low and high threshold.
    :type low_high: np.array with shape (2,) with dtype = np.float32

    :param auto_threshold: Use automatic thresholding
    :type auto_threshold: bool

    :return: Edge image with values between 0 and 1 if hysteresis is skipped otherwise either 0 or 1.
    :rtype: np.array with shape (height, width) with dtype = np.floating
    """

    # TODO: check if a single kernel is faster or coordinate multiple kernel calls and used them for better sync

    # Disable warnings about low gpu utilization in the test suite
    old_cuda_low_occupancy_warnings = nb.config.CUDA_LOW_OCCUPANCY_WARNINGS
    nb.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

    stream = cuda.stream()

    height, width = image_u8_i.shape
    blockspergrid_x = math.ceil(height / TPB)
    blockspergrid_y = math.ceil(width / TPB)
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Check if the image buffer needs to be copied to the device
    input_on_device = is_cuda_array(image_u8_i)
    d_image_u8_i = (
        image_u8_i if input_on_device else cuda.to_device(image_u8_i, stream=stream)
    )

    # Convert input image to floating
    # Allocate a new array to store the floating image
    d_image_i = cuda.device_array((height, width), dtype=np.float32, stream=stream)
    _kernel_convert_to_float32[blockspergrid, (TPB, TPB), stream](
        d_image_u8_i, d_image_i
    )

    # Allocate output arrays
    d_blurred = cuda.device_array((height, width), dtype=np.float32, stream=stream)

    # Apply Gaussian filter
    # We store the blurred image in the gradients array
    _kernel_gauss[blockspergrid, (TPB, TPB), stream](d_image_i, d_blurred, sigma)

    # Allocate output arrays
    d_gradients = d_image_i  # Reuse the image buffer for the gradients
    d_image_i = None
    d_orientations = cuda.device_array((height, width), dtype=np.float32, stream=stream)

    # Compute the sobel gradient
    _kernel_gradient_sobel[blockspergrid, (TPB, TPB), stream](
        d_blurred, d_gradients, d_orientations
    )

    # Apply Non-Maxima Suppression
    _kernel_non_max[blockspergrid, (TPB, TPB), stream](d_gradients, d_orientations)

    # Compute the auto thresholds
    if auto_threshold:
        low, high = _kernel_compute_hysteresis_auto_thresholds[
            blockspergrid, (TPB, TPB), stream
        ](d_gradients, low, high)

    # Apply hysteresis thresholding
    _kernel_hysteresis[blockspergrid, (TPB, TPB), stream](d_gradients, low, high)

    edges = d_gradients if input_on_device else d_gradients.copy_to_host(stream=stream)

    # Reset warnings
    nb.config.CUDA_LOW_OCCUPANCY_WARNINGS = old_cuda_low_occupancy_warnings

    stream.synchronize()

    return edges


implementation_options = AttrDict({})

implementation_metadata = AttrDict(
    {
        "display_name": "Numba CUDA fp32",
        "type": "cuda",
    }
)
