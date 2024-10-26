"""
Canny end-to-end implementation using just numpy and scipy.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import scipy.signal as signal
from utils.attr_dict import AttrDict
from icecream import ic


def _convert_to_float(image_u8_i: np.array) -> np.array:
    if image_u8_i.dtype != np.float32:
        return image_u8_i.astype(np.float32) / 255.0
    return image_u8_i


def gaussian_kernel(sigma: float) -> np.array:
    # kernel_width = 2 * one_dir + 1
    kernel_width_half = np.ceil(3 * sigma)

    # Create a 2D grid of indices
    idx = np.arange(-kernel_width_half, kernel_width_half + 1)
    x, y = np.meshgrid(idx, idx)

    # Compute the Gaussian kernel values
    kernel = 1 / (2 * np.pi * (sigma**2)) * (np.exp(-(x**2 + y**2) / (2 * (sigma**2))))

    return kernel


def _blur_gauss(image_i: np.array, sigma: float) -> np.array:
    kernel = gaussian_kernel(sigma)

    # TODO: write a numpy only convolution using sliding_window_view and using just indices
    img_blur_o = signal.convolve2d(image_i, kernel, mode="same", boundary="symm")
    # TODO: compare with: img_blur_o = cv2.filter2D(src=image_i, ddepth=-1, kernel=kernel)
    return img_blur_o


def blur_gauss(image_u8_i: np.array, sigma: float) -> np.array:
    """Apply a Gaussian blur to the input image.

    :param image_u8_i: Input image in grayscale
    :type image_u8_i: np.array with shape (height, width) with dtype = np.uint8

    :param sigma: Standard deviation of the gaussian filter
    :type sigma: float

    :return: Blurred image
    :rtype: np.array with shape (height, width) with dtype = np.floating and values in the range [0., 1.]
    """

    image_f = _convert_to_float(image_u8_i)
    image_blurred = _blur_gauss(image_f, sigma)
    return image_blurred


def _sobel(image_i: np.array, prewitt: bool = False) -> tuple[np.array, np.array]:
    """Apply the Sobel filter to the input image and return the gradient and the orientation.

    :param image_i: Grayscale input image
    :type image_i: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param prewitt: Use the Prewitt operator instead of the Sobel operator
    :type prewitt: bool

    :return: (gradient, orientation): gradient: edge strength of the image in range [0.,1.],
                                      orientation: angle of gradient in range [-np.pi, np.pi]
    :rtype: (np.array, np.array)
    """

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.transpose(sobel_x)

    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.transpose(prewitt_x)

    kernel_x = prewitt_x if prewitt else sobel_x
    kernel_y = prewitt_y if prewitt else sobel_y

    # TODO: impl using only numpy
    # TODO: speed compare cv2, scipy, numpy
    # import cv2
    # img_x = cv2.filter2D(src=image_i, ddepth=-1, kernel=kernel_x)
    # img_y = cv2.filter2D(src=image_i, ddepth=-1, kernel=kernel_y)

    img_x = -signal.convolve2d(image_i, kernel_x, boundary="symm", mode="same")
    img_y = -signal.convolve2d(image_i, kernel_y, boundary="symm", mode="same")

    # Clipping is necessary on some machines to avoid going over 1.0
    gradient_o = np.clip(np.sqrt(img_x**2 + img_y**2), 0.0, 1.0)
    orientation_o = np.atan2(img_y, img_x)

    return gradient_o, orientation_o


def sobel_gradients(image_i: np.array) -> tuple[np.array, np.array]:
    """Compute the gradients and orientations of the input image using the Sobel operator.

    :param image_i: Input image in grayscale this array will be overwritten with the gradients
    :type image_i: np.array with shape (height, width) with dtype = np.floating and values in the range [0., 1.]

    :return: (gradient, orientation): gradient: edge strength of the image in range [0.,1.],
                                      orientation: angle of gradient in range [-np.pi, np.pi]
    :rtype: (np.array, np.array)
    """

    gradient, orientation = _sobel(image_i)
    return gradient, orientation


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
    # This non_max suppression implementation uses no loops and no branching.
    # It is very fast and works great on CPUs.

    # How it works:
    # 1. Pad the gradients array with zeros
    # 2. Create a sliding window view of the padded gradients array such that we can access the neighbours of each pixel by its own index.
    # 3. Discretise the orientations into 8 sectors and mask the correct pairs of neighbours.
    # 4. Apply the mask to the edges array
    # 5. Done

    height, width = gradients_i.shape
    gs = (height, width)

    # Pad the gradients array with zeros.
    gradients_padded = np.zeros((height + 2, width + 2), dtype=gradients_i.dtype)
    gradients_padded[1:-1, 1:-1] = gradients_i

    # Prepare the offset grids for vector manipulation.
    window = sliding_window_view(gradients_padded, gs, writeable=True)

    # x=-1,y=0
    # shift right
    gradient_left = window[1][0]
    # x=1, y=0
    # shift left
    gradient_right = window[1][2]
    # x=-1,y=-1
    # shift down right
    gradient_top_left = window[0][0]
    # x=1, y=1
    # shift up left
    gradient_bottom_right = window[2][2]
    # x=0,y=-1
    # shift down
    gradient_top = window[0][1]
    # x=0, y=1
    # shift up
    gradient_bottom = window[2][1]
    # x=1, y=-1
    # shift down left
    gradient_top_right = window[0][2]
    # x=-1, y=1
    # shift up right
    gradient_bottom_left = window[2][0]

    # Computing the neighbour masks is rather simple.
    # Each mask is responsible for a pair of neighbours, and two opposite orientation sectors.

    # Compute neighbour mask: left, right sector
    # should be called mask_horizontal but we call it just mask because we immediately reuse the array
    mask = (
        (orientations_i <= -7 / 8 * np.pi)  # left
        | (orientations_i > 7 / 8 * np.pi)  # left
        | (
            (orientations_i > -1 / 8 * np.pi)  # right
            & (orientations_i <= 1 / 8 * np.pi)  # right
        )
    )
    # note optimized by using &=
    mask &= (gradients_i >= gradient_right) & (gradients_i > gradient_left)
    # Compute neighbour mask: top right, bottom left sector
    mask_diag_1_tr_bl = (
        (
            (orientations_i > 1 / 8 * np.pi)  # top right
            & (orientations_i <= 3 / 8 * np.pi)
        )  # top right
        | (
            (orientations_i > -7 / 8 * np.pi)  # bottom left
            & (orientations_i <= -5 / 8 * np.pi)
        )  # bottom left
    )
    mask |= (
        mask_diag_1_tr_bl
        & (gradients_i > gradient_top_left)
        & (gradients_i > gradient_bottom_right)
    )
    # Compute neighbour mask: top, bottom sector
    mask_vertical = (
        (
            (orientations_i > 3 / 8 * np.pi)  # top
            & (orientations_i <= 5 / 8 * np.pi)
        )  # top
        | (
            (orientations_i > -5 / 8 * np.pi)  # bottom
            & (orientations_i <= -3 / 8 * np.pi)
        )  # bottom
    )
    mask |= (
        mask_vertical & (gradients_i > gradient_top) & (gradients_i >= gradient_bottom)
    )
    # Compute neighbour mask: top left, bottom right sector
    mask_diag_2_tl_br = (
        (
            (orientations_i > 5 / 8 * np.pi)  # top left
            & (orientations_i <= 7 / 8 * np.pi)
        )  # top left
        | (
            (orientations_i > -3 / 8 * np.pi)  # bottom right
            & (orientations_i <= -1 / 8 * np.pi)
        )  # bottom right
    )
    mask |= (
        mask_diag_2_tl_br
        & (gradients_i > gradient_top_right)
        & (gradients_i > gradient_bottom_left)
    )

    edges = window[1][1]
    # Apply the mask to the edges
    edges *= mask

    return edges


HISTOGRAM_BIN_COUNT = 256


def _compute_hysteresis_auto_thresholds(
    edges_i: np.array, low_prop: float, high_prop: float
) -> tuple[float, float]:
    """Compute the hysteresis thresholds based on the gradient strength.

    :param gradients_i: Edge strength of the image in range [0.,1.]
    :type gradients_i: np.array

    :param low_prop: Proportion of the lowest gradient values to be used as the low threshold
    :type low_prop: float

    :param high_prop: Proportion of the highest gradient values to be used as the high threshold
    :type high_prop: float

    :return: (low, high): low: Low threshold for the hysteresis, high: High threshold for the hysteresis
    :rtype: (float, float)
    """

    # Compute a histogram of the edges
    histogram, _ = np.histogram(edges_i, bins=HISTOGRAM_BIN_COUNT, range=(0.0, 1.0))
    # Convert the histogram to a cumulative histogram and discard the first bucket.
    # The first bucket contains all the pixels that have already been filtered by non_max and canny.
    cumulative_histogram = np.cumsum(histogram[1:])

    # The last bucket of the cumulative histogram contains the number of the remaining pixels.
    total_pixels = cumulative_histogram[-1]

    # The cumulative histogram is sorted so we can use searchsorted to find the bucket index for the thresholds.
    low_threshold_idx = np.searchsorted(
        cumulative_histogram, (1.0 - low_prop) * total_pixels, side="right"
    )
    # We exploit the fact that high_prop > low_prop and only search in the remaining buckets.
    high_threshold_idx = (
        np.searchsorted(
            cumulative_histogram[low_threshold_idx:],
            (1.0 - high_prop) * total_pixels,
            side="right",
        )
        + low_threshold_idx
    )

    # +1 because we skipped the first bucket
    low = float(low_threshold_idx + 1) / HISTOGRAM_BIN_COUNT
    high = float(high_threshold_idx + 1) / HISTOGRAM_BIN_COUNT

    return low, high


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

    low, high = _compute_hysteresis_auto_thresholds(
        gradients_i, low_high_prop_i[0], low_high_prop_i[1]
    )
    return [low, high]


def _hysteresis(edges_i: np.array, low: float, high: float) -> np.array:
    # This hysteresis implementation uses only one loop and _no_ branching.
    # Unfortunately, it is very slow.
    # The same method works great on GPUs though.
    # This was my first draft for the GPU implementation.

    # How it works:
    # 1. Snap all values that are above the high threshold to 1.0
    # 2. Pad the edges array with zeros
    # 3. Create a sliding window view of the padded edges array such that we can access the neighbours of each pixel by its own index.
    # 4. Create a mask that allows weak edges to remain and be extended if they are connected to a strong edge.
    # 5. Apply the mask to the edges array
    # 6. Now we loop until the mask no longer changes. This signals that the expansion has converged.
    # 7. Finally, we discard all values that are not 1.0
    # 8. Done

    height, width = edges_i.shape
    gs = (height, width)

    # TODO: for end-to-end this internal function should reuse the non_max stuff

    # Pad the gradients array with zeros.
    gradients_padded = np.zeros((height + 2, width + 2), dtype=edges_i.dtype)
    gradients_padded[1:-1, 1:-1] = edges_i

    # TODO: check if its faster to do with the window[1][1]
    # Snap all values that are above the high threshold to 1.0.
    mask_high = gradients_padded >= high
    np.maximum(gradients_padded, mask_high, out=gradients_padded)

    # Prepare the offset grids for vector manipulation.
    window = sliding_window_view(gradients_padded, gs, writeable=True)

    edges = window[1][1]

    # x=-1,y=0
    # shift right
    edge_left = window[1][0]
    # x=1, y=0
    # shift left
    edge_right = window[1][2]
    # x=-1,y=-1
    # shift down right
    edge_top_left = window[0][0]
    # x=1, y=1
    # shift up left
    edge_bottom_right = window[2][2]
    # x=0,y=-1
    # shift down
    edge_top = window[0][1]
    # x=0, y=1
    # shift up
    edge_bottom = window[2][1]
    # x=1, y=-1
    # shift down left
    edge_top_right = window[0][2]
    # x=-1, y=1
    # shift up right
    edge_bottom_left = window[2][0]

    mask_weak_edges = edges >= low

    # Compute the initial neighbour mask.
    mask = mask_weak_edges & (
        (edge_left == 1.0)
        | (edge_right == 1.0)
        | (edge_top_left == 1.0)
        | (edge_bottom_right == 1.0)
        | (edge_top == 1.0)
        | (edge_bottom == 1.0)
        | (edge_top_right == 1.0)
        | (edge_bottom_left == 1.0)
    )
    last_extended_edge_count = 0
    current_extended_edge_count = np.any(mask)
    # ic(current_extended_edge_count)

    # Loop until the mask no longer changes.
    while last_extended_edge_count != current_extended_edge_count:
        last_extended_edge_count = current_extended_edge_count
        # Apply the mask to the edges.
        np.maximum(edges, mask, out=edges)
        # Update the mask.
        mask = mask_weak_edges & (
            (edge_left == 1.0)
            | (edge_right == 1.0)
            | (edge_top_left == 1.0)
            | (edge_bottom_right == 1.0)
            | (edge_top == 1.0)
            | (edge_bottom == 1.0)
            | (edge_top_right == 1.0)
            | (edge_bottom_left == 1.0)
        )
        current_extended_edge_count = np.sum(mask)
        # ic(np.sum(mask))

    # only keep values that are 1.0
    mask = edges == 1.0
    edges *= mask

    return edges


def hysteresis(gradients_i: np.array, low_high_thresholds_i: np.array) -> np.array:
    """Apply hysteresis to the input gradients image.

    :param gradients_i: Edge strength of the image in range [0.,1.]
    :type gradients_i: np.array

    :param low_high_thresholds_i: Array with the low and high threshold for the hysteresis
    :type low_high_thresholds_i: np.array with shape (2,) with dtype = np.float32

    :return: Edge image with values either 0 or 1
    :rtype: np.array with shape (height, width) with dtype = np.floating
    """

    edges = _hysteresis(gradients_i, low_high_thresholds_i[0], low_high_thresholds_i[1])
    return edges


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

    image_blurred = blur_gauss(image_u8_i, sigma)
    gradients, orientations = sobel_gradients(image_blurred)
    edges = non_max(gradients, orientations)

    if auto_threshold:
        low, high = _compute_hysteresis_auto_thresholds(
            gradients, low_high_i[0], low_high_i[1]
        )

    edges = _hysteresis(edges, low, high)

    return edges


implementation_options = AttrDict({})

implementation_metadata = AttrDict(
    {
        "display_name": "Numpy vec v4 dibit",
        "type": "cpu",
    }
)
