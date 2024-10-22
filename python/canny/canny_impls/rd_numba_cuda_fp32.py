"""
Canny end-to-end implementation using numba and cuda.
"""

# TODO: create a version with fp16 https://numba.readthedocs.io/en/stable/cuda-reference/kernel.html#bit-floating-point-intrinsics

import numpy as np
import numba as nb
from numba import cuda
import math
from utils.attr_dict import AttrDict
from numba.types import void, float32, uint8

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


@cuda.jit(fastmath=True, func_or_sig=void(float32[:, :], float32), device=True)
def _dev_gauss_inplace(image_io: np.array, sigma: float):
    x, y = cuda.grid(2)

    # compute the kernel and store it in the shared memory
    kernel_width = int(2.0 * math.ceil(3.0 * sigma) + 1.0)
    # TODO: The kernel should fit into a local array.
    # TODO: Check if it is faster if each thread computes the kernel on its own.
    # TODO: Check if its faster if the kernel is stored in constant memory or passed as an argument.
    # kernel = cuda.shared.array(shape=(kernel_width, kernel_width), dtype=np.float32)
    kernel = cuda.shared.array(0, dtype=nb.types.float32)
    if x < kernel_width and y < kernel_width:
        kernel_val = (
            1.0
            / (2.0 * np.pi * (sigma**2.0))
            * (math.exp(-(x**2.0 + y**2.0) / (2.0 * (sigma**2.0))))
        )
        kernel[x + y * kernel_width] = kernel_val
    cuda.syncthreads()

    x_width, y_height = image_io.shape[0], image_io.shape[1]
    one_dir = int(math.ceil(3.0 * sigma))

    if x < x_width and y < y_height:
        # compute the convolution
        result = 0.0
        for i in range(-one_dir, one_dir):
            for j in range(-one_dir, one_dir):
                x_i = x + i
                y_j = y + j
                if x_i >= 0 and x_i < x_width and y_j >= 0 and y_j < y_height:
                    kernel_val = (
                        kernel[(i + one_dir) + (j + one_dir) * kernel_width]
                        * image_io[x_i, y_j]
                    )
                    result += kernel_val
        cuda.syncthreads()
        image_io[x, y] = result


# TODO: check if its faster to use a two arrays instead of the io parameter
@cuda.jit(fastmath=True, func_or_sig=void(float32[:, :], float32))
def _kernel_gauss_inplace(image_io: np.array, sigma: float):
    _dev_gauss_inplace(image_io, sigma)


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
    if not input_on_device:
        image_u8_i = cuda.to_device(image_u8_i, stream=stream)

    # Convert input image to floating
    d_image = cuda.device_array((height, width), dtype=np.float32, stream=stream)
    _kernel_convert_to_float32[blockspergrid, (TPB, TPB), stream](image_u8_i, d_image)

    # Apply Gaussian filter
    _kernel_gauss_inplace[blockspergrid, (TPB, TPB), stream](d_image, sigma)

    blurred = d_image if input_on_device else d_image.copy_to_host(stream=stream)

    # Reset warnings
    nb.config.CUDA_LOW_OCCUPANCY_WARNINGS = old_cuda_low_occupancy_warnings

    stream.synchronize()

    return blurred


# TODO: check if its faster to use a three arrays instead of the io parameter
@cuda.jit(fastmath=True, func_or_sig=void(float32[:, :], float32[:, :]), device=True)
def _dev_gradient_sobel(gradients_io: np.array, orientations_o: np.array):
    x, y = cuda.grid(2)

    x_width, y_height = (
        gradients_io.shape[0],
        gradients_io.shape[1],
    )

    if x < x_width and y < y_height:
        # Compute the gradient
        dx = 0.0

        x_left_idx = x - 1
        x_left = 0.0
        if x_left_idx >= 0:
            x_left = gradients_io[x_left_idx, y]

        x_right_idx = x + 1
        x_right = 0.0
        if x_right_idx < x_width:
            x_right = gradients_io[x_right_idx, y]

        dx = x_right - 2.0 * gradients_io[x, y] + x_left

        dy = 0.0

        y_top_idx = y - 1
        y_top = 0.0
        if y_top_idx >= 0:
            y_top = gradients_io[x, y_top_idx]

        y_bottom_idx = y + 1
        y_bottom = 0.0
        if y_bottom_idx < y_height:
            y_bottom = gradients_io[x, y_bottom_idx]

        dy = y_bottom - 2.0 * gradients_io[x, y] + y_top

        orientations_o[x, y] = np.arctan2(dy, dx)

        cuda.syncthreads()
        gradients_io[x, y] = math.sqrt(dx**2 + dy**2)


@cuda.jit(fastmath=True, func_or_sig=void(float32[:, :], float32[:, :]))
def _kernel_gradient_sobel(gradients_io: np.array, orientations_o: np.array):
    _dev_gradient_sobel(gradients_io, orientations_o)


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
    if not input_on_device:
        d_gradients_io = cuda.to_device(image_i, stream=stream)
    else:
        # Allocate a new array to store the gradients
        d_gradients_io = cuda.device_array(
            (height, width), dtype=np.float32, stream=stream
        )
        # Copy the input image to the new array
        _kernel_copy_devarray2d[blockspergrid, (TPB, TPB), stream](
            image_i, d_gradients_io
        )

    d_orientations = cuda.device_array((height, width), dtype=np.float32, stream=stream)

    _kernel_gradient_sobel[blockspergrid, (TPB, TPB), stream](
        d_gradients_io, d_orientations
    )

    gradients = (
        d_gradients_io
        if input_on_device
        else d_gradients_io.copy_to_host(stream=stream)
    )
    orientations = (
        d_orientations
        if input_on_device
        else d_orientations.copy_to_host(stream=stream)
    )

    # Reset warnings
    nb.config.CUDA_LOW_OCCUPANCY_WARNINGS = old_cuda_low_occupancy_warnings

    stream.synchronize()

    return gradients, orientations


# TODO: check if its faster to use a three arrays instead of the io parameter
@cuda.jit(fastmath=True, device=True, func_or_sig=void(float32[:, :], float32[:, :]))
def _dev_non_max(gradients_io, orientations_o):
    x, y = cuda.grid(2)
    x_width, y_height = gradients_io.shape[0], gradients_io.shape[1]

    if x >= x_width or y >= y_height:
        return

    orientation = orientations_o[x, y]
    gradient = gradients_io[x, y]

    # Compute neighbour mask: top, bottom sector
    mask_vertical = (
        ((orientation > 3 / 8 * np.pi) & (orientation <= 5 / 8 * np.pi))  # top
        | ((orientation > -5 / 8 * np.pi) & (orientation <= -3 / 8 * np.pi))  # bottom
    )
    gradient_right = 0
    if x + 1 < x_width:
        gradient_right = gradients_io[x + 1, y]
    gradient_left = 0
    if x - 1 >= 0:
        gradient_left = gradients_io[x - 1, y]
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
        gradient_top_left = gradients_io[x - 1, y - 1]
    gradient_bottom_right = 0
    if x + 1 < x_width and y + 1 < y_height:
        gradient_bottom_right = gradients_io[x + 1, y + 1]
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
        gradient_top = gradients_io[x, y - 1]
    gradient_bottom = 0
    if y + 1 < y_height:
        gradient_bottom = gradients_io[x, y + 1]
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
        gradient_top_right = gradients_io[x + 1, y - 1]
    gradient_bottom_left = 0
    if x - 1 >= 0 and y + 1 < y_height:
        gradient_bottom_left = gradients_io[x - 1, y + 1]
    mask |= (
        mask_diag_2_tl_br
        & (gradient > gradient_top_right)  # top right
        & (gradient > gradient_bottom_left)  # bottom left
    )

    gradients_io[x, y] = mask * gradient


@cuda.jit(fastmath=True, func_or_sig=void(float32[:, :], float32[:, :]))
def _kernel_non_max(gradients_io: np.array, orientations_o: np.array):
    _dev_non_max(gradients_io, orientations_o)


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
    gradients_on_device = is_cuda_array(gradients_i)
    orientations_on_device = is_cuda_array(orientations_i)

    if not gradients_on_device:
        d_gradients = cuda.to_device(gradients_i, stream=stream)
    else:
        # Allocate a new array to store the gradients
        d_gradients = cuda.device_array(
            (height, width), dtype=np.float32, stream=stream
        )
        # Copy the input image to the new array
        _kernel_copy_devarray2d[blockspergrid, (TPB, TPB), stream](
            gradients_i, d_gradients
        )

    d_orientations = (
        orientations_i
        if orientations_on_device
        else cuda.to_device(orientations_i, stream=stream)
    )

    _kernel_non_max[blockspergrid, (TPB, TPB), stream](d_gradients, d_orientations)

    gradients = (
        d_gradients if gradients_on_device else d_gradients.copy_to_host(stream=stream)
    )

    # Reset warnings
    nb.config.CUDA_LOW_OCCUPANCY_WARNINGS = old_cuda_low_occupancy_warnings

    stream.synchronize()

    return gradients


HISTOGRAM_BIN_COUNT = 256


@cuda.jit(fastmath=True, device=True, func_or_sig=void(float32[:, :], float32[:]))
def _dev_compute_hysteresis_auto_thresholds(
    gradients_i: np.array, low_high_thresholds_io: np.array
) -> tuple[float, float]:
    x, y = cuda.grid(2)
    x_width, y_height = gradients_i.shape[0], gradients_i.shape[1]

    if x >= x_width or y >= y_height:
        return

    gradient = gradients_i[x, y]

    # Compute the histogram of the gradients
    histogram = cuda.shared.array(HISTOGRAM_BIN_COUNT, float32)

    histogram_bin_width = 1.0 / HISTOGRAM_BIN_COUNT
    bin_idx = int(gradient / histogram_bin_width)
    if bin_idx >= HISTOGRAM_BIN_COUNT:
        bin_idx = HISTOGRAM_BIN_COUNT - 1
    cuda.atomic.add(histogram, bin_idx, 1)

    cuda.syncthreads()
    # TODO: check if its better if all threads but one return early here

    low_prop = low_high_thresholds_io[0]
    high_prop = low_high_thresholds_io[1]

    total_pixels = x_width * y_height
    low_pixels = total_pixels * low_prop
    high_pixels = total_pixels * high_prop

    pixel_bucket_sum = 0
    low_threshold = 0.0
    high_threshold = 0.0
    # cumsum but skip the first bin because it contains all the zero pixels
    for i in range(1, HISTOGRAM_BIN_COUNT):
        pixel_bucket_sum += histogram[i]
        if low_threshold == 0.0 and pixel_bucket_sum >= low_pixels:
            low_threshold = i * histogram_bin_width
        if pixel_bucket_sum >= high_pixels:
            high_threshold = i * histogram_bin_width
            break

    # TODO: check if its better if only one thread writes the result
    low_high_thresholds_io[0] = low_threshold
    low_high_thresholds_io[1] = high_threshold


@cuda.jit(fastmath=True, func_or_sig=void(float32[:, :], float32[:]))
def _kernel_compute_hysteresis_auto_thresholds(
    gradients_i: np.array, low_high_thresholds_io: np.array
) -> tuple[float, float]:
    _dev_compute_hysteresis_auto_thresholds(gradients_i, low_high_thresholds_io)


@cuda.jit(fastmath=True, device=True, func_or_sig=void(float32[:, :], float32, float32))
def _dev_hysteresis(gradients_io, low_threshold, high_threshold):
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
    x_width, y_height = gradients_io.shape[0], gradients_io.shape[1]

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
    low: float,
    high: float,
    auto_threshold: bool = False,
) -> np.array:
    """Apply Canny edge detection to the input image.

    :param image_u8_i: Input image in grayscale
    :type image_u8_i: np.array with shape (height, width) with dtype = np.uint8

    :param sigma: Standard deviation of the gaussian filter
    :type sigma: float

    :param low: Low threshold for the hysteresis
    :type low: float

    :param high: High threshold for the hysteresis
    :type high: float

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
    if not input_on_device:
        image_u8_i = cuda.to_device(image_u8_i, stream=stream)

    # Convert input image to floating
    d_gradients = cuda.device_array((height, width), dtype=np.float32, stream=stream)
    _kernel_convert_to_float32[blockspergrid, (TPB, TPB), stream](
        image_u8_i, d_gradients
    )

    # Allocate output arrays
    d_orientations = cuda.device_array((height, width), dtype=np.float32, stream=stream)

    # Apply Gaussian filter
    # We store the blurred image in the gradients array
    _kernel_gauss_inplace[blockspergrid, (TPB, TPB), stream](d_gradients, sigma)

    # Compute gradient
    _kernel_gradient_sobel[blockspergrid, (TPB, TPB), stream](
        d_gradients, d_orientations
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
