import cv2
import numpy as np

from icecream import ic

import utils.distinct_colors as distinct_colors


def draw_keypoints(image_i: np.array, keypoints: list) -> np.array:
    image_o = image_i.copy()

    # if it is grayscale convert to color
    if len(image_o.shape) == 2:
        image_o = cv2.cvtColor(image_o, cv2.COLOR_GRAY2RGB)

    max_size = max(image_o.shape)
    thickness = int(max(np.round(max_size / 300), 1))

    # For subpixel accuracy
    shift_bits = 4
    shift_multiplier = 1 << 4

    color_iter = distinct_colors.bgrs()
    for key_point in keypoints:
        color = next(color_iter)
        color = tuple(map(lambda x: x * 255, color))

        center = (
            int(np.round(key_point.pt[0] * shift_multiplier)),
            int(np.round(key_point.pt[1] * shift_multiplier)),
        )
        radius = int(np.round(key_point.size / 2 * shift_multiplier))
        cv2.circle(
            image_o,
            center,
            radius,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
            shift=shift_bits,
        )
        if key_point.angle != -1:
            srcAngleRad = key_point.angle * np.pi / 180.0
            orient = (
                int(np.round(np.cos(srcAngleRad) * radius)),
                int(np.round(np.sin(srcAngleRad) * radius)),
            )
            cv2.line(
                image_o,
                center,
                (center[0] + orient[0], center[1] + orient[1]),
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
                shift=shift_bits,
            )

    return image_o


def draw_matches(
    image_1_i: np.array,
    keypoints_1: list,
    image_2_i: np.array,
    keypoints_2: list,
    matches: list,
) -> np.array:
    max_size = max(*image_1_i.shape, *image_2_i.shape)
    thickness = int(max(np.round(max_size / 300), 1))

    return cv2.drawMatches(
        image_1_i,
        keypoints_1,
        image_2_i,
        keypoints_2,
        matches,
        None,
        matchesThickness=thickness,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
