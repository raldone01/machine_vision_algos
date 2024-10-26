"""
Canny end-to-end implementation using numba and cuda.
"""

# TODO: create a version with fp16 https://numba.readthedocs.io/en/stable/cuda-reference/kernel.html#bit-floating-point-intrinsics

import numpy as np
import numba as nb
from numba import cuda
import math
from utils.attr_dict import AttrDict
from numba.types import void, float32, uint8, uint32, int32
from icecream import ic

TPB = 16
TPB_PLUS_TWO = TPB + 2


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


MAX_2D_DIM_F32_SHARED_ARRAY = 110


@cuda.jit(fastmath=True, func_or_sig=void(float32[:, :], float32[:, :], float32))
def _kernel_gauss(image_i: np.array, image_o: np.array, sigma: float):
    x_width, y_height = image_i.shape

    # pixel coordinates
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # local thread coordinates
    l_x = cuda.threadIdx.x
    l_y = cuda.threadIdx.y

    l_x_size = cuda.blockDim.x
    l_y_size = cuda.blockDim.y

    # compute half of the kernel width
    kernel_width_half = int(math.ceil(3.0 * sigma)) + 1
    kernel = cuda.shared.array(
        (MAX_2D_DIM_F32_SHARED_ARRAY, MAX_2D_DIM_F32_SHARED_ARRAY),
        dtype=nb.types.float32,
    )

    factor = 2.0 * math.pi * (sigma**2.0)

    # Compute one quarter of the kernel and store it in shared memory
    for i in range(l_x, kernel_width_half, l_x_size):
        for j in range(l_y, kernel_width_half, l_y_size):
            kernel_val = (math.exp(-(i**2.0 + j**2.0) / (2.0 * (sigma**2.0)))) / factor
            kernel[i, j] = kernel_val
    cuda.syncthreads()

    # Only execute if within image bounds
    x_width, y_height = image_i.shape
    if not (x < x_width and y < y_height):
        return

    # TODO: the 4 quadrant computations can be optimized to be done in parallel
    result = 0.0
    # First Quadrant
    for i in range(0, kernel_width_half):
        for j in range(0, kernel_width_half):
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
            kernel_val = kernel[i, j] * image_i[x_i, y_j]
            result += kernel_val
    # Second Quadrant
    for i in range(1, kernel_width_half):
        for j in range(1, kernel_width_half):
            x_i = x - i
            y_j = y - j
            if x_i < 0:
                x_i = 0
            if x_i >= x_width:
                x_i = x_width - 1
            if y_j < 0:
                y_j = 0
            if y_j >= y_height:
                y_j = y_height - 1
            kernel_val = kernel[i, j] * image_i[x_i, y_j]
            result += kernel_val
    # Third Quadrant
    for i in range(1, kernel_width_half):
        for j in range(0, kernel_width_half):
            x_i = x - i
            y_j = y + j
            if x_i < 0:
                x_i = 0
            if x_i >= x_width:
                x_i = x_width - 1
            if y_j < 0:
                y_j = 0
            if y_j >= y_height:
                y_j = y_height - 1
            kernel_val = kernel[i, j] * image_i[x_i, y_j]
            result += kernel_val
    # Fourth Quadrant
    for i in range(0, kernel_width_half):
        for j in range(1, kernel_width_half):
            x_i = x + i
            y_j = y - j
            if x_i < 0:
                x_i = 0
            if x_i >= x_width:
                x_i = x_width - 1
            if y_j < 0:
                y_j = 0
            if y_j >= y_height:
                y_j = y_height - 1
            kernel_val = kernel[i, j] * image_i[x_i, y_j]
            result += kernel_val

    # Write back the result to the image
    # image_o[x, y] = result

    plot_kernel = False
    if plot_kernel:
        image_o[x, y] = result
        # Dump the kernel in the top left corner of the image
        if (
            x < MAX_2D_DIM_F32_SHARED_ARRAY
            and y < MAX_2D_DIM_F32_SHARED_ARRAY
            and x < x_width
            and y < y_height
        ):
            image_o[x, y] = kernel[x, y] * factor
    else:
        image_o[x, y] = result


def check_gauss_kernel_size(sigma: float):
    kernel_width_half = int(math.ceil(3.0 * sigma)) + 1
    kernel_width = 2 * math.ceil(3 * sigma) + 1
    if kernel_width_half > MAX_2D_DIM_F32_SHARED_ARRAY:
        max_supported_width = MAX_2D_DIM_F32_SHARED_ARRAY * 2 - 1
        raise ValueError(
            f"Kernel width {kernel_width} is too large. Maximum supported width is {max_supported_width}."
        )


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
    check_gauss_kernel_size(sigma)
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
HISTOGRAM_BIN_COUNT_PLUS_ONE = HISTOGRAM_BIN_COUNT + 1


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

    # Initialize shared memory (also the one at the end that is used to store the final histogram)
    shared = cuda.shared.array(
        shape=HISTOGRAM_BIN_COUNT_PLUS_ONE, dtype=nb.types.uint32
    )
    # Initialize the histogram bins to zero
    for i in range(linear_tid, HISTOGRAM_BIN_COUNT_PLUS_ONE, BLOCK_THREADS):
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

    x_partial_histogram_count = (
        partial_histograms_i.shape[0] // HISTOGRAM_BIN_COUNT
    ) - 1

    # NOTE: These are essentially HISTOGRAM_BIN_COUNT parallel cumulative sums
    # https://people.cs.vt.edu/yongcao/teaching/cs5234/spring2013/slides/Lecture10.pdf
    # https://en.wikipedia.org/wiki/Prefix_sum
    # https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

    # Sum all the partial histograms
    final_histogram_offset = HISTOGRAM_BIN_COUNT * x_partial_histogram_count
    if x < HISTOGRAM_BIN_COUNT:
        total = 0
        for i in range(x_partial_histogram_count):
            total += partial_histograms_i[x + i * HISTOGRAM_BIN_COUNT]
        partial_histograms_i[final_histogram_offset + x] = total

    # All threads return early except the first block
    if cuda.blockIdx.x != 0:
        return

    cuda.syncthreads()

    # Calculate the final cumulative histogram
    histogram_cumulative = cuda.shared.array(
        shape=HISTOGRAM_BIN_COUNT, dtype=nb.types.uint32
    )
    if x < HISTOGRAM_BIN_COUNT:
        total = 0
        # The first bucket is empty anyway
        for i in range(1, x + 1):
            total += partial_histograms_i[final_histogram_offset + i]
        histogram_cumulative[x] = total

    cuda.syncthreads()

    low_prop, high_prop = low_high_prop_i[0], low_high_prop_i[1]

    total_pixels = histogram_cumulative[HISTOGRAM_BIN_COUNT - 1]
    low_pixels = total_pixels * (1.0 - low_prop)
    high_pixels = total_pixels * (1.0 - high_prop)

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
            if next_bucket >= low_pixels:
                low_high_thresholds_o[0] = (x + 1) / HISTOGRAM_BIN_COUNT
        if bucket <= high_pixels:
            # We might have found the high threshold.
            # Check if the next bucket is above the high threshold.
            next_bucket_idx = x + 1
            if next_bucket_idx < HISTOGRAM_BIN_COUNT:
                next_bucket = histogram_cumulative[next_bucket_idx]
            else:
                next_bucket = 0xFFFFFFFF
            if next_bucket >= high_pixels:
                low_high_thresholds_o[1] = (x + 1) / HISTOGRAM_BIN_COUNT


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

    # ic(histogram_count)
    # The one at the end is used to store the final histogram
    histogram_count += 1

    d_partial_histograms = cuda.device_array(
        histogram_count * HISTOGRAM_BIN_COUNT,
        dtype=np.uint32,
        stream=stream,
    )

    # Compute the auto thresholds
    _kernel_compute_edge_histogram_partial[blockspergrid, (TPB, TPB), stream](
        d_gradients_i, d_partial_histograms
    )

    TPB_FA = HISTOGRAM_BIN_COUNT
    blockspergrid_x = math.ceil(histogram_count / TPB_FA)
    _kernel_compute_edge_histogram_final_accum[blockspergrid_x, TPB_FA, stream](
        d_partial_histograms, d_low_high_prop_i, d_low_high_thresholds_o
    )

    low_high_thresholds_o = (
        d_low_high_thresholds_o
        if input_on_device
        else d_low_high_thresholds_o.copy_to_host(stream=stream)
    )

    # Reset warnings
    nb.config.CUDA_LOW_OCCUPANCY_WARNINGS = old_cuda_low_occupancy_warnings

    stream.synchronize()

    return low_high_thresholds_o


@cuda.jit(
    fastmath=True,
    func_or_sig=void(float32[:, :], float32[:], int32[:], float32[:, :], int32[:]),
)
def _kernel_hysteresis(
    edges_i: np.array,
    low_high_thresholds_i: np.array,
    pixel_offset_i: np.array,
    edges_o: np.array,
    blocks_that_found_a_new_edge_o: np.array,
):
    """
    Apply hysteresis thresholding to the gradients array.

    1) DEV: Copy gradients for one block into shared memory (including one pixel border).
    2) DEV: Expand the strong edges to the weak edges.
    3) DEV: Atomic flag signals that the block is done.
    4) DEV: Copy the block back to global memory (excluding the border).
    5) DEV: Each block updates the change flag if it found new edges.
    6) HOST: If the change flag is set we need to iterate again with the pixel_offset=(-BLOCK/2, -BLOCK/2).
             One block more is required because of the offset grid!
    """

    x_width, y_height = edges_i.shape

    x_offset, y_offset = pixel_offset_i[0], pixel_offset_i[1]

    low, high = low_high_thresholds_i[0], low_high_thresholds_i[1]
    # DEBUG
    # low = 0.75
    # high = 0.95

    # pixel coordinates
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # grid dimensions
    nx = cuda.gridDim.x * cuda.blockDim.x
    ny = cuda.gridDim.y * cuda.blockDim.y

    # local block coordinates
    l_x = cuda.threadIdx.x
    l_y = cuda.threadIdx.y

    # linear thread index within 2D block
    linear_tid = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x

    # total number of threads in the 2D block
    BLOCK_THREADS = cuda.blockDim.x * cuda.blockDim.y

    # linear block index within 2D grid
    g = cuda.blockIdx.x + cuda.blockIdx.y * cuda.gridDim.x
    g_x_width = cuda.gridDim.x
    g_y_height = cuda.gridDim.y
    gx = cuda.blockIdx.x
    gy = cuda.blockIdx.y
    gox = g_x_width * gx
    goy = g_y_height * gy

    # Initialize shared memory
    block_cache = cuda.shared.array(
        shape=(TPB_PLUS_TWO, TPB_PLUS_TWO), dtype=nb.types.float32
    )

    # DEBUG INIT SHARED MEMORY
    for six in range(TPB_PLUS_TWO):
        for siy in range(TPB_PLUS_TWO):
            block_cache[six, siy] = -0.6
    cuda.syncthreads()

    # 1) Copy from edges_i to shared memory
    edges_x = x - x_offset
    edges_y = y - y_offset
    if edges_x >= 0 and edges_x < x_width and edges_y >= 0 and edges_y < y_height:
        block_cache[l_x + 1, l_y + 1] = edges_i[edges_x, edges_y]
    else:
        block_cache[l_x + 1, l_y + 1] = 0.0
    # Copy the border pixels without the corners
    # left border
    if l_x == 0:
        x_left = edges_x - 1
        if edges_y >= 0 and edges_y < y_height and x_left >= 0 and x_left < x_width:
            block_cache[0, l_y + 1] = edges_i[x_left, edges_y]
        else:
            block_cache[0, l_y + 1] = 0.0
    # right border
    if l_x == TPB - 1:
        x_right = edges_x + 1
        if edges_y >= 0 and edges_y < y_height and x_right >= 0 and x_right < x_width:
            block_cache[TPB + 1, l_y + 1] = edges_i[x_right, edges_y]
        else:
            block_cache[TPB + 1, l_y + 1] = 0.0
    # top border
    if l_y == 0:
        y_top = edges_y - 1
        if edges_x >= 0 and edges_x < x_width and y_top >= 0 and y_top < y_height:
            block_cache[l_x + 1, 0] = edges_i[edges_x, y_top]
        else:
            block_cache[l_x + 1, 0] = 0.0
    # bottom border
    if l_y == TPB - 1:
        y_bottom = edges_y + 1
        if edges_x >= 0 and edges_x < x_width and y_bottom >= 0 and y_bottom < y_height:
            block_cache[l_x + 1, TPB + 1] = edges_i[edges_x, y_bottom]
        else:
            block_cache[l_x + 1, TPB + 1] = 0.0
    # Copy the corners
    # top left
    if l_x == 0 and l_y == 0:
        x_left = edges_x - 1
        y_top = edges_y - 1
        if x_left >= 0 and x_left < x_width and y_top >= 0 and y_top < y_height:
            block_cache[0, 0] = edges_i[x_left, y_top]
        else:
            block_cache[0, 0] = 0.0
    # top right
    if l_x == TPB - 1 and l_y == 0:
        x_right = edges_x + 1
        y_top = edges_y - 1
        if x_right >= 0 and x_right < x_width and y_top >= 0 and y_top < y_height:
            block_cache[TPB + 1, 0] = edges_i[x_right, y_top]
        else:
            block_cache[TPB + 1, 0] = 0.0
    # bottom left
    if l_x == 0 and l_y == TPB - 1:
        x_left = edges_x - 1
        y_bottom = edges_y + 1
        if x_left >= 0 and x_left < x_width and y_bottom >= 0 and y_bottom < y_height:
            block_cache[0, TPB + 1] = edges_i[x_left, y_bottom]
        else:
            block_cache[0, TPB + 1] = 0.0
    # bottom right
    if l_x == TPB - 1 and l_y == TPB - 1:
        x_right = edges_x + 1
        y_bottom = edges_y + 1
        if x_right >= 0 and x_right < x_width and y_bottom >= 0 and y_bottom < y_height:
            block_cache[TPB + 1, TPB + 1] = edges_i[x_right, y_bottom]
        else:
            block_cache[TPB + 1, TPB + 1] = 0.0
    cuda.syncthreads()

    # 2) Expand the strong edges to the weak edges
    # Borders can be ignored because they are handled by other thread blocks

    lo_x = l_x + 1
    lo_y = l_y + 1

    is_discard_or_strong = False
    # 2.1) Snap all edges above the high threshold to 1.0
    edge = block_cache[lo_x, lo_y]
    if edge >= high:
        block_cache[lo_x, lo_y] = 1.0
        is_discard_or_strong = True
    # 2.2) Discard all edges below the low threshold
    elif edge < low:
        block_cache[lo_x, lo_y] = 0.0
        is_discard_or_strong = True

    # 3) Expand the strong edges to the weak edges
    found_new_edge_this_iteration = 1
    found_new_edge = False
    while cuda.syncthreads_or(found_new_edge_this_iteration):
        found_new_edge_this_iteration = 0
        if not is_discard_or_strong:
            # 3.1) Check if the weak edge has a strong edge neighbour
            # top left
            found_new_edge_this_iteration |= block_cache[lo_x - 1, lo_y - 1] == 1.0
            # top
            found_new_edge_this_iteration |= block_cache[lo_x, lo_y - 1] == 1.0
            # top right
            found_new_edge_this_iteration |= block_cache[lo_x + 1, lo_y - 1] == 1.0
            # right
            found_new_edge_this_iteration |= block_cache[lo_x + 1, lo_y] == 1.0
            # bottom right
            found_new_edge_this_iteration |= block_cache[lo_x + 1, lo_y + 1] == 1.0
            # bottom
            found_new_edge_this_iteration |= block_cache[lo_x, lo_y + 1] == 1.0
            # bottom left
            found_new_edge_this_iteration |= block_cache[lo_x - 1, lo_y + 1] == 1.0
            # left
            found_new_edge_this_iteration |= block_cache[lo_x - 1, lo_y] == 1.0

            if found_new_edge_this_iteration:
                # print("FNE", x, y, gx, gy)
                block_cache[lo_x, lo_y] = 1.0
                is_discard_or_strong = True
                found_new_edge = True

    any_new_edge_found = cuda.syncthreads_or(int(found_new_edge))

    if l_x == 0 and l_y == 0:
        # print("ANE", any_new_edge_found, gx, gy)
        if any_new_edge_found:
            cuda.atomic.add(blocks_that_found_a_new_edge_o, 0, 1)

    # 4) Copy the block back to global memory
    if edges_x >= 0 and edges_x < x_width and edges_y >= 0 and edges_y < y_height:
        edges_o[edges_x, edges_y] = block_cache[l_x + 1, l_y + 1]

    return
    # START DEBUG SINGLE SHARED MEM BLOCK
    # copy the first block of shared memory to global memory
    cuda.syncthreads()
    if cuda.blockIdx.x == 1 and cuda.blockIdx.y == 0 and l_x == 0 and l_y == 0:
        for six in range(0, x_width):
            for siy in range(0, y_height):
                edges_o[six, siy] = -0.5

        for six in range(0, TPB_PLUS_TWO):
            for siy in range(0, TPB_PLUS_TWO):
                edges_o[six, siy] = block_cache[six, siy]
    return
    # END DEBUG SINGLE SHARED MEM BLOCK


@cuda.jit(
    fastmath=True,
    func_or_sig=void(float32[:, :], float32[:, :]),
)
def _kernel_hysteresis_final_filter(
    edges_i: np.array,
    edges_o: np.array,
):
    x, y = cuda.grid(2)
    x_width, y_height = edges_i.shape

    if x >= x_width or y >= y_height:
        return

    if edges_i[x, y] == 1.0:
        val = 1.0
    else:
        val = 0.0
    edges_o[x, y] = val


def hysteresis(
    gradients_i: np.array, low_high_thresholds_i: np.array, stream_i=None
) -> np.array:
    """Apply hysteresis to the input gradients image.

    :param gradients_i: Edge strength of the image in range [0.,1.]
    :type gradients_i: np.array

    :param low_high_thresholds_i: Array with the low and high threshold for the hysteresis
    :type low_high_thresholds_i: np.array with shape (2,) with dtype = np.float32

    :return: Edge image with values either 0 or 1
    :rtype: np.array with shape (height, width) with dtype = np.floating
    """

    # Disable warnings about low gpu utilization in the test suite
    old_cuda_low_occupancy_warnings = nb.config.CUDA_LOW_OCCUPANCY_WARNINGS
    nb.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

    stream = cuda.stream() if stream_i is None else stream_i

    height, width = gradients_i.shape
    blockspergrid_x = math.ceil(height / TPB)
    blockspergrid_y = math.ceil(width / TPB)
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Check if the image buffer needs to be copied to the device
    gradients_on_device = is_cuda_array(gradients_i)
    d_gradients_i = (
        gradients_i
        if gradients_on_device
        else cuda.to_device(gradients_i, stream=stream)
    )
    low_high_thresholds_on_device = is_cuda_array(low_high_thresholds_i)
    d_low_high_thresholds_i = (
        low_high_thresholds_i
        if low_high_thresholds_on_device
        else cuda.to_device(low_high_thresholds_i, stream=stream)
    )
    any_input_on_device = gradients_on_device or low_high_thresholds_on_device

    # Allocate arrays
    d_edges_o = cuda.device_array((height, width), dtype=np.float32, stream=stream)

    # DEBUG INIT OUTPUT
    debug_d_edges_o = cuda.to_device(np.full_like(gradients_i, -1.1), stream=stream)
    _kernel_copy_devarray2d[blockspergrid, (TPB, TPB), stream](
        debug_d_edges_o,
        d_edges_o,
    )

    d_blocks_that_found_a_new_edge = cuda.to_device(
        np.array([0], dtype=np.int32), stream=stream
    )

    pixel_offset_shift = TPB // 2
    pixel_offset = np.array([0, 0], dtype=np.int32)
    # pixel_offset = np.array([5, 5], dtype=np.int32)
    d_pixel_offset = cuda.to_device(pixel_offset, stream=stream)

    # Apply hysteresis thresholding

    last_found_edge_counter = 0
    current_found_edge_counter = 0
    any_new_edge_found = True
    it = 0
    while any_new_edge_found:
        # +1 to accommodate the grid offset
        blockspergrid = (blockspergrid_x + 1, blockspergrid_y + 1)
        _kernel_hysteresis[blockspergrid, (TPB, TPB), stream](
            d_gradients_i,
            d_low_high_thresholds_i,
            d_pixel_offset,
            d_edges_o,
            d_blocks_that_found_a_new_edge,
        )
        # swap d_gradients_i and d_edges_o
        d_gradients_i, d_edges_o = d_edges_o, d_gradients_i

        last_found_edge_counter = current_found_edge_counter
        current_found_edge_counter = d_blocks_that_found_a_new_edge.copy_to_host(
            stream=stream
        )[0]
        stream.synchronize()
        it += 1

        # TODO: at the end of the kernel this can be done on the gpu
        if it % 2 != 0:
            pixel_offset[0] = pixel_offset_shift
            pixel_offset[1] = pixel_offset_shift
        else:
            pixel_offset[0] = 0
            pixel_offset[1] = 0
        any_new_edge_found = current_found_edge_counter - last_found_edge_counter > 0
        if any_new_edge_found:
            cuda.to_device(pixel_offset, to=d_pixel_offset, stream=stream)
        # ic(it, current_found_edge_counter, pixel_offset)
        # DEBUG
        # any_new_edge_found = False

    # Filter the final image
    _kernel_hysteresis_final_filter[blockspergrid, (TPB, TPB), stream](
        d_gradients_i,
        d_edges_o,
    )

    edges_o = (
        d_edges_o if any_input_on_device else d_edges_o.copy_to_host(stream=stream)
    )

    # Reset warnings
    nb.config.CUDA_LOW_OCCUPANCY_WARNINGS = old_cuda_low_occupancy_warnings

    if stream_i is None:
        stream.synchronize()

    return edges_o


def canny_edge_detection(
    image_u8_i: np.array,
    sigma: float,
    low_high_i: np.array,
    auto_threshold: bool = False,
) -> np.array:
    """Apply Canny edge detection to the input image.

    :param image_u8_i: Input image in grayscale
    :type image_u8_i: np.array with shape (height, width) with dtype = np.uint8

    :param sigma: Standard deviation of the gaussian filter
    :type sigma: float

    :param low_high_i: Array with the low and high threshold for the hysteresis. If auto_threshold is True this array will be used as the proportion of the lowest and highest gradient values to be used as the low and high threshold.
    :type low_high_i: np.array with shape (2,) with dtype = np.float32

    :param auto_threshold: Use automatic thresholding
    :type auto_threshold: bool

    :return: Edge image with values between 0 and 1 if hysteresis is skipped otherwise either 0 or 1.
    :rtype: np.array with shape (height, width) with dtype = np.floating
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

    # Compute gaussian blur

    # Convert input image to floating
    # Allocate a new array to store the floating image
    d_image_i = cuda.device_array(image_u8_i.shape, dtype=np.float32, stream=stream)
    _kernel_convert_to_float32[blockspergrid, (TPB, TPB), stream](
        d_image_u8_i, d_image_i
    )

    # Allocate output array
    d_blurred = cuda.device_array(image_u8_i.shape, dtype=np.float32, stream=stream)
    # Apply Gaussian filter
    check_gauss_kernel_size(sigma)
    _kernel_gauss[blockspergrid, (TPB, TPB), stream](d_image_i, d_blurred, sigma)

    # Compute sobel gradients

    # Allocate output arrays
    d_gradients = cuda.device_array((height, width), dtype=np.float32, stream=stream)
    d_orientations = cuda.device_array((height, width), dtype=np.float32, stream=stream)

    _kernel_gradient_sobel[blockspergrid, (TPB, TPB), stream](
        d_blurred, d_gradients, d_orientations
    )

    # Apply Non-Maxima Suppression

    # Allocate output array
    # Reuse the blurred array to store the edges
    d_edges = d_blurred
    d_blurred = None

    _kernel_non_max[blockspergrid, (TPB, TPB), stream](
        d_gradients, d_orientations, d_edges
    )

    # Hysteresis thresholding

    d_low_high_thresholds_i = (
        low_high_i
        if is_cuda_array(low_high_i)
        else cuda.to_device(low_high_i, stream=stream)
    )

    if auto_threshold:
        # Allocate output arrays
        d_low_high_thresholds_o = cuda.device_array(2, dtype=np.float32, stream=stream)
        histogram_count = blockspergrid_x * blockspergrid_y
        # The one at the end is used to store the final histogram
        histogram_count += 1

        d_partial_histograms = cuda.device_array(
            histogram_count * HISTOGRAM_BIN_COUNT,
            dtype=np.uint32,
            stream=stream,
        )

        _kernel_compute_edge_histogram_partial[blockspergrid, (TPB, TPB), stream](
            d_edges, d_partial_histograms
        )

        TPB_FA = HISTOGRAM_BIN_COUNT
        blockspergrid_x = math.ceil(histogram_count / TPB_FA)
        _kernel_compute_edge_histogram_final_accum[blockspergrid_x, TPB_FA, stream](
            d_partial_histograms, d_low_high_thresholds_i, d_low_high_thresholds_o
        )
    else:
        d_low_high_thresholds_o = d_low_high_thresholds_i

    d_edges = hysteresis(d_edges, d_low_high_thresholds_o)

    edges = d_edges.copy_to_host(stream=stream) if not input_on_device else d_edges

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
