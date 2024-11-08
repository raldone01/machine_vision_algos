import numpy as np
import cv2

from utils.plotting_tools import SmartFigure, plot_matrix, plot_kernel
from IPython.display import display
from icecream import ic
from numpy.lib.stride_tricks import sliding_window_view


def _convert_to_float(image_u8_i: np.array) -> np.array:
    if image_u8_i.dtype != np.float32:
        return image_u8_i.astype(np.float32) / 255.0
    return image_u8_i


def _sigma_to_kernel_size(sigma: float) -> int:
    kernel_width_half = np.ceil(3 * sigma)
    kernel_width = 2 * kernel_width_half + 1
    return int(kernel_width)


def _new_gaussian_kernel(sigma: float) -> np.array:
    kernel_width = _sigma_to_kernel_size(sigma)
    kernel = cv2.getGaussianKernel(kernel_width, sigma)
    return np.outer(kernel, kernel.transpose())


def _non_max(input_array: np.array) -> np.array:
    """Return a matrix in which only local maxima of the input mat are set to True, all other values are False

    :param mat: Input matrix
    :type mat: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range (-inf, 1.]

    :return: Binary Matrix with the same dimensions as the input matrix
    :rtype: np.ndarray with shape (height, width) with dtype = bool
    """

    # Initialize a 3x3 kernel with ones and a zero in the middle
    kernel = np.ones(shape=(3, 3), dtype=np.uint8)
    kernel[1, 1] = 0

    # Apply the OpenCV dilate morphology transformation.
    # For details see https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    dilation = cv2.dilate(input_array, kernel)
    return input_array > dilation


def _harris_corner(
    image_u8_i: np.array, sigma1: float, sigma2: float, k: float, threshold: float
) -> list[cv2.KeyPoint]:
    """Detect corners using the Harris corner detector

    In this function, corners in a grayscale image are detected using the Harris corner detector.
    They are returned in a list of OpenCV KeyPoints (https://docs.opencv.org/4.x/d2/d29/classcv_1_1KeyPoint.html).
    Each KeyPoint includes the attributes, pt (position), size, angle, response. The attributes size and angle are not
    relevant for the Harris corner detector and can be set to an arbitrary value. The response is the result of the
    Harris corner formula.

    :param image_u8_i: Input image in grayscale
    :type image_u8_i: np.array with shape (height, width) with dtype = np.uint8

    :param sigma1: Sigma for the first Gaussian filtering
    :type sigma1: float

    :param sigma2: Sigma for the second Gaussian filtering
    :type sigma2: float

    :param k: Coefficient for harris formula
    :type k: float

    :param threshold: corner threshold
    :type threshold: float

    :return: keypoints:
        corners: List of cv2.KeyPoints containing all detected corners after thresholding and non-maxima suppression.
            Each keypoint has the attribute pt[x, y], size, angle, response.
                pt: The x, y position of the detected corner in the OpenCV coordinate convention.
                size: The size of the relevant region around the keypoint. Not relevant for Harris and is set to 1.
                angle: The direction of the gradient in degree. Relative to image coordinate system (clockwise).
                response: Result of the Harris corner formula R = det(M) - k*trace(M)**2
    :rtype: List[cv2.KeyPoint]

    """
    keypoints = []

    image_f = _convert_to_float(image_u8_i)

    height, width = image_f.shape

    gauss_1_kernel = _new_gaussian_kernel(sigma1)
    gauss_2_kernel = _new_gaussian_kernel(sigma2)

    gauss_1_kernel_y, gauss_1_kernel_x = np.gradient(gauss_1_kernel)

    """
    debug_sf = SmartFigure()
    debug_fig = debug_sf.get_fig()

    ax = []
    ax.append(debug_fig.add_subplot(2, 2, 1, projection="3d"))
    plot_kernel(ax[0], gauss_1_kernel, "Gaussian Kernel 1")
    ax.append(debug_fig.add_subplot(2, 2, 2, projection="3d"))
    plot_kernel(ax[1], gauss_2_kernel, "Gaussian Kernel 2")

    ax.append(debug_fig.add_subplot(2, 2, 3, projection="3d"))
    plot_kernel(ax[2], gauss_1_kernel_x, "Gaussian Kernel 1 X Gradient")
    ax.append(debug_fig.add_subplot(2, 2, 4, projection="3d"))
    plot_kernel(ax[3], gauss_1_kernel_y, "Gaussian Kernel 1 Y Gradient")

    debug_sf.display_as_image()
    """

    i_x = cv2.filter2D(src=image_f, ddepth=-1, kernel=gauss_1_kernel_x)
    i_y = cv2.filter2D(src=image_f, ddepth=-1, kernel=gauss_1_kernel_y)
    i_xy = i_x * i_y

    # weigh the gradients by gauss_2_kernel
    i_x = cv2.filter2D(src=i_x, ddepth=-1, kernel=gauss_2_kernel)
    i_y = cv2.filter2D(src=i_y, ddepth=-1, kernel=gauss_2_kernel)
    i_xy = cv2.filter2D(src=i_xy, ddepth=-1, kernel=gauss_2_kernel)

    M_matrices = np.stack([i_x**2, i_xy, i_xy, i_y**2], axis=-1).reshape(
        height, width, 2, 2
    )

    # Compute the corner response
    det_M = np.linalg.det(M_matrices)
    trace_M = np.trace(M_matrices, axis1=2, axis2=3)
    response = det_M - k * trace_M**2

    # Normalize the response to 1
    response = response / np.max(response)

    # Threshold the response
    response_thresholded_mask = response > threshold
    response_thresholded = response * response_thresholded_mask

    # Apply non-maxima suppression
    response_thresholded = _non_max(response_thresholded)

    # Create a list of keypoints
    for x in range(width):
        for y in range(height):
            if response_thresholded[y, x]:
                keypoint = cv2.KeyPoint(x, y, 1, -1, response[y, x])
                keypoints.append(keypoint)

    return keypoints


def harris_corner(
    image_set_u8_i: np.array, sigma1: float, sigma2: float, k: float, threshold: float
) -> list[list[cv2.KeyPoint]]:
    """Detect corners using the Harris corner detector for a set of images

    In this function, corners in a set of grayscale images are detected using the Harris corner detector.
    They are returned in a list of lists of OpenCV KeyPoints (https://docs.opencv.org/4.x/d2/d29/classcv_1_1KeyPoint.html).
    Each KeyPoint includes the attributes, pt (position), size, angle, response. The attributes size and angle are not
    relevant for the Harris corner detector and can be set to an arbitrary value. The response is the result of the
    Harris corner formula.

    :param image_set_u8_i: Input images in grayscale
    :type image_set_u8_i: np.array with shape (num_images, height, width) with dtype = np.uint8

    :param sigma1: Sigma for the first Gaussian filtering
    :type sigma1: float

    :param sigma2: Sigma for the second Gaussian filtering
    :type sigma2: float

    :param k: Coefficient for harris formula
    :type k: float

    :param threshold: corner threshold
    :type threshold: float

    :return: keypoints:
        corners: List of cv2.KeyPoints containing all detected corners after thresholding and non-maxima suppression.
            Each keypoint has the attribute pt[x, y], size, angle, response.
                pt: The x, y position of the detected corner in the OpenCV coordinate convention.
                size: The size of the relevant region around the keypoint. Not relevant for Harris and is set to 1.
                angle: The direction of the gradient in degree. Relative to image coordinate system (clockwise).
                response: Result of the Harris corner formula R = det(M) - k*trace(M)**2
    :rtype: List[List[cv2.KeyPoint]]

    """
    keypoints = []
    for image_u8_i in image_set_u8_i:
        keypoints.append(_harris_corner(image_u8_i, sigma1, sigma2, k, threshold))
    return keypoints


def compute_descriptors(
    image_i: np.ndarray, keypoints_i: list[cv2.KeyPoint], patch_size: int
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Calculate a descriptor on patches of the image, centred on the locations of the KeyPoints.

    Calculate a descriptor vector for each keypoint in the list. KeyPoints that are too close to the border to include
    the whole patch are filtered out. The descriptors are returned as a k x m matrix with k being the number of filtered
    KeyPoints and m being the length of a descriptor vector (patch_size**2). The descriptor at row i of
    the descriptors array is the descriptor for the KeyPoint filtered_keypoint[i].

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param keypoints: List of keypoints at which to compute the descriptors
    :type keypoints: List[cv2.KeyPoint]

    :param patch_size: Value defining the width and height of the patch around each keypoint to calculate descriptor.
    :type patch_size: int

    :return: (filtered_keypoints, descriptors):
        filtered_keypoints: List of the filtered keypoints.
            Locations too close to the image boundary to cut out the image patch should not be contained.
        descriptors: k x m matrix containing the patch descriptors.
            Each row vector stores the descriptor vector of the respective corner.
            with k being the number of descriptors and m being the length of a descriptor (usually patch_size**2).
            The descriptor at row i belongs to the KeyPoint at filtered_keypoints[i]
    :rtype: (List[cv2.KeyPoint], np.ndarray)
    """
    window = sliding_window_view(image_i, (patch_size, patch_size))

    keypoints_coords = np.array([kp.pt for kp in keypoints_i], dtype=np.int32)
    keypoints_x, keypoints_y = keypoints_coords[:, 0], keypoints_coords[:, 1]

    # compute the window indices for the keypoints
    patch_size_half = patch_size // 2
    keypoints_window_x = keypoints_x - patch_size_half
    keypoints_window_y = keypoints_y - patch_size_half

    # compute the keypoint mask
    # left
    keypoint_mask = keypoints_window_x >= 0
    # right
    keypoint_mask &= keypoints_window_x < window.shape[1]
    # top
    keypoint_mask &= keypoints_window_y >= 0
    # bottom
    keypoint_mask &= keypoints_window_y < window.shape[0]

    # filter out the keypoints that are too close to the border
    descriptors = window[
        keypoints_window_y[keypoint_mask], keypoints_window_x[keypoint_mask]
    ].reshape(-1, patch_size**2)
    filtered_keypoints = [kp for kp, mask in zip(keypoints_i, keypoint_mask) if mask]

    # assert that the shape of the descriptors is correct
    assert descriptors.shape == (len(filtered_keypoints), patch_size**2)

    # sort the values in the descriptors
    descriptors = np.sort(descriptors, axis=1)  # seems to work a little better

    # normalize the descriptors
    # descriptors = descriptors - np.max(descriptors, axis=1)[:, None]
    # descriptors = descriptors / np.linalg.norm(descriptors, axis=1)[:, None]

    return filtered_keypoints, descriptors


def flann_matches(
    descriptors1: np.ndarray, descriptors2: np.ndarray
) -> list[cv2.DMatch]:
    # FLANN (Fast Library for Approximate Nearest Neighbors) parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(
        descriptors1.astype(np.float32), descriptors2.astype(np.float32), k=2
    )
    return matches


def filter_matches(matches: tuple[tuple[cv2.DMatch]]) -> list[cv2.DMatch]:
    """Filter out all matches that do not satisfy the Lowe Distance Ratio Condition

    :param matches: Holds all the possible matches. Each 'row' are matches of one source_keypoint to target_keypoint
    :type matches: Tuple of tuples of cv2.DMatch https://docs.opencv.org/master/d4/de0/classcv_1_1DMatch.html

    :return filtered_matches: A list of all matches that fulfill the Low Distance Ratio Condition
    :rtype: List[cv2.DMatch]
    """
    # d_a = np.array([m[0].distance for m in matches])
    # d_b = np.array([m[1].distance for m in matches])

    matches = np.array(matches)
    d_a, d_b = np.array([(m[0].distance, m[1].distance) for m in matches]).T

    # perform the Lowe Distance Ratio Test
    ratio = 0.8
    mask = d_a < ratio * d_b

    filtered_matches = matches[mask, 0]

    return filtered_matches


def find_homography_leastsquares(
    source_points: np.ndarray,
    target_points: np.ndarray,
    confidence: float = None,
    inlier_threshold: float = None,
) -> np.ndarray:
    """Return projective transformation matrix of source_points in the target image given matching points

    Return the projective transformation matrix for homogeneous coordinates. It uses the Least-Squares algorithm to
    minimize the back-projection error with all points provided. Requires at least 4 matching points.

    :param source_points: Array of points. Each row holds one point from the source image (object image) as [x, y]
    :type source_points: np.ndarray with shape (n, 2)

    :param target_points: Array of points. Each row holds one point from the target image (scene image) as [x, y].
    :type target_points: np.ndarray with shape (n, 2)

    :return: The projective transformation matrix for homogeneous coordinates with shape (3, 3)
    :rtype: np.ndarray with shape (3, 3)
    """
    assert source_points.shape == target_points.shape

    n = source_points.shape[0]
    assert n >= 4

    # create the A matrix
    A = np.empty((2 * n, 8), dtype=np.float64)
    for i in range(n):
        x, y = source_points[i]
        x_p, y_p = target_points[i]
        A[2 * i] = np.array([x, y, 1, 0, 0, 0, -x * x_p, -y * x_p])
        A[2 * i + 1] = np.array([0, 0, 0, x, y, 1, -x * y_p, -y * y_p])

    # solve with np.linalg.lstsq
    b = target_points.flatten()
    homography_reduced = np.linalg.lstsq(A, b, rcond=None)[0]

    # we need to add back h22
    homography = np.append(homography_reduced, 1).reshape(3, 3)

    return homography


def find_homography_ransac(
    source_points: np.ndarray,
    target_points: np.ndarray,
    confidence: float,
    inlier_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return estimated transformation matrix of source_points in the target image given matching points

    Return the projective transformation matrix for homogeneous coordinates. It uses the RANSAC algorithm with the
    Least-Squares algorithm to minimize the back-projection error and be robust against outliers.
    Requires at least 4 matching points.

    :param source_points: Array of points. Each row holds one point from the source (object) image [x, y]
    :type source_points: np.ndarray with shape (n, 2)

    :param target_points: Array of points. Each row holds one point from the target (scene) image [x, y].
    :type target_points: np.ndarray with shape (n, 2)

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers.
    :type confidence: float

    :param inlier_threshold: Max. Euclidean distance of a point from the transformed point to be considered an inlier
    :type inlier_threshold: float

    :return: (homography, inliers, num_iterations)
        homography: The projective transformation matrix for homogeneous coordinates with shape (3, 3)
        inliers: Is True if the point at the index is an inlier. Boolean array with shape (n,)
        num_iterations: The number of iterations that were needed for the sample consensus
    :rtype: Tuple[np.ndarray, np.ndarray, int]
    """
    best_suggested_homography = np.eye(3)
    best_inliers = np.full(shape=len(target_points), fill_value=True, dtype=bool)
    num_iterations = 0

    return best_suggested_homography, best_inliers, num_iterations
