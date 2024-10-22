import cv2
import numpy as np


def add_gaussian_noise(
    image_i: np.array, mean: float = 0.0, sigma: float = 0.1
) -> np.array:
    """Applies additive Gaussian noise to the input grayscale image

    :param img: Input grayscale image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param mean: Mean of the Gaussian distribution the noise is drawn from
    :type mean: float

    :param sigma: Standard deviation of the Gaussian distribution the noise is drawn from
    :type sigma: float

    :return: Image with added noise
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    """
    noisy_img = image_i.copy()

    noise = sigma * np.random.randn(*noisy_img.shape) + mean
    noisy_img += noise
    cv2.normalize(noisy_img, noisy_img, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    # Sometimes values are outside of limits due to rounding errors. Cut those values:
    np.clip(noisy_img, 0.0, 1.0, out=noisy_img)
    return noisy_img
