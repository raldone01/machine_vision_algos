import os
from pathlib import Path
from unittest import mock

from ipywidgets import Layout

import cv2

from utils.benchmarking import LogTimer


def load_input_images(input_images_folder) -> list:
    input_image_files = [
        f
        for f in os.listdir(input_images_folder)
        if f.endswith(".jpg") or f.endswith(".png")
    ]
    input_image_files.sort()

    input_images = []

    with LogTimer(f"Loading {len(input_image_files)} images"):
        for image_filename in input_image_files:
            with LogTimer(f"Loading {image_filename}"):
                input_image = mock.Mock()

                input_image.filename = image_filename
                input_image.filepath = Path(input_images_folder) / image_filename

                input_image.image_color = cv2.imread(
                    str(input_image.filepath), cv2.IMREAD_COLOR
                )
                input_image.image_gray = cv2.cvtColor(
                    input_image.image_color, cv2.COLOR_BGR2GRAY
                )

                input_images.append(input_image)

    # sort the image list by size
    input_images.sort(key=lambda image: image.image_color.size)
    return input_images
