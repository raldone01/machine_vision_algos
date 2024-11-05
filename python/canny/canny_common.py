import os
from pathlib import Path

from utils.benchmarking import LogTimer
from utils.image_tools import load_image


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
            input_images.append(load_image(Path(input_images_folder, image_filename)))

    # sort the image list by size
    input_images.sort(key=lambda image: image.image_color.size)
    return input_images
