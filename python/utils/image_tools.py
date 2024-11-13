from dataclasses import astuple, dataclass
from pathlib import Path

import cv2
import numpy as np

from utils.benchmarking import LogTimer


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
    noise = sigma * np.random.randn(*image_i.shape) + mean
    noisy_img = image_i + noise
    cv2.normalize(noisy_img, noisy_img, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    # Sometimes values are outside of limits due to rounding errors. Cut those values:
    np.clip(noisy_img, 0.0, 1.0, out=noisy_img)
    return noisy_img


@dataclass
class LoadedImage:
    filename: str = ""
    filepath: Path = None
    image_color: np.array = None
    image_gray: np.array = None

    def __iter__(self):
        return iter(astuple(self))

    def __str__(self):
        return (
            f"LoadedImage(filename={self.filename}, "
            f"filepath={self.filepath}, "
            f"image_color.shape={self.image_color.shape}, "
            f"image_gray.shape={self.image_gray.shape})"
        )


def load_image(image_path: Path) -> LoadedImage:
    """Loads an image from the specified path and returns it as a LoadedImage object

    :param image_path: Path to the image file
    :type image_path: Path

    :return: LoadedImage object with the loaded image
    :rtype: LoadedImage
    """

    # Convert image_path to a Path object if it is a string
    if isinstance(image_path, str):
        image_path = Path(image_path)

    with LogTimer(f"Loading {image_path.name}"):
        image = LoadedImage()
        image.filename = image_path.name
        image.filepath = image_path
        image.image_color = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image.image_gray = cv2.cvtColor(image.image_color, cv2.COLOR_BGR2GRAY)
    return image


def save_image(image_buf_i: np.ndarray, file_path: Path):
    """Saves the image to the specified path

    :param image: LoadedImage object with the image to be saved
    :type image: LoadedImage

    :param file_path: Path to save the image to
    :type file_path: Path
    """

    if isinstance(file_path, str):
        file_path = Path(file_path)

    with LogTimer(f"Saving {file_path.name}"):
        cv2.imwrite(str(file_path), image_buf_i)
    return
