from io import BytesIO
import logging

import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image as PILImage
from IPython.display import Image
from ipywidgets import widgets

open_cnt = 0


class SmartFigure:
    """
    Raii wrapper for matplotlib figures.
    It closes the figure when the object is destroyed.
    """

    def __init__(self, figsize=(14, 14), dpi=100, rc_params=None, **kwargs):
        self._logger = logging.Logger("SmartFigure")

        rc_params_default = plt.rcParams.copy()

        # https://stackoverflow.com/a/39566040/4479969
        SMALL_SIZE = 22
        MEDIUM_SIZE = 26
        BIGGER_SIZE = 60

        rc_params_better = {
            "figure.titlesize": BIGGER_SIZE,
            "font.size": SMALL_SIZE,
            "axes.titlesize": BIGGER_SIZE,
            "axes.labelsize": MEDIUM_SIZE,
            "xtick.labelsize": SMALL_SIZE,
            "ytick.labelsize": SMALL_SIZE,
            "legend.fontsize": SMALL_SIZE,
        }

        rc_params_default.update(rc_params_better)

        if rc_params is not None:
            rc_params_default.update(rc_params)

        with plt.ioff():
            if rc_params is not None:
                with plt.rc_context(rc_params):
                    self.fig = plt.figure(figsize=figsize, dpi=dpi, **kwargs)
            else:
                self.fig = plt.figure(figsize=figsize, dpi=dpi, **kwargs)

        # Minor ticks can't be styled in the style-sheet
        for ax in self.fig.axes:
            ax.grid(visible=True, which="both", axis="both")
            ax.grid(
                visible=True, which="major", color="dimgray", linestyle="-", alpha=0.9
            )
            ax.grid(
                visible=True, which="minor", color="silver", linestyle="-", alpha=0.5
            )
            ax.minorticks_on()
            # ax.ticklabel_format(style='sci', useMathText=True)
            # ax.tick_params(axis='x', labelrotation = -45)

        global open_cnt
        open_cnt += 1
        self._logger.debug("Opening figure", self.fig, open_cnt)

    def get_fig(self):
        return self.fig

    def __del__(self):
        global open_cnt
        self._logger.debug("Closing figure", self.fig, open_cnt)
        plt.close(self.fig)


def print_open_plot_count():
    open_plots_count = len(plt.get_fignums())
    print(f"There are {open_plots_count} open plots.")


def _prepare_image_buf(image_buf_i: np.array, longest_side: int = None):
    if len(image_buf_i.shape) == 3:
        is_color = True
    elif len(image_buf_i.shape) == 2:
        is_color = False
    else:
        raise ValueError(
            "The image does not have a valid shape. Expected either (height, width) or (height, width, channels)"
        )

    if image_buf_i.dtype == np.uint8:
        image_buf_o = image_buf_i.astype(np.float32) / 255.0
    else:
        image_buf_o = image_buf_i.astype(np.float32)

    height, width = image_buf_o.shape[:2]
    if longest_side is not None and (height > longest_side or width > longest_side):
        scale = longest_side / max(height, width)
        cv2.resize(
            image_buf_o,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
            dst=image_buf_o,
        )

    if is_color:
        cv2.cvtColor(image_buf_o, cv2.COLOR_BGR2RGB, dst=image_buf_o)
    else:
        cv2.cvtColor(image_buf_o, cv2.COLOR_GRAY2RGB, dst=image_buf_o)

    return image_buf_o, is_color


def plot_image(ax, image_buf_i, longest_side=None):
    image_buf_o, is_color = _prepare_image_buf(image_buf_i, longest_side)

    if is_color:
        ax.imshow(image_buf_o)
    else:
        ax.imshow(image_buf_o, cmap="gray")
    ax.axis("off")
    return ax, image_buf_o


def to_ipy_image(image_buf_i, fmt="png", longest_side=None, use_widget=True):
    image_buf_o, is_color = _prepare_image_buf(image_buf_i, longest_side)

    image_buf_o = (image_buf_o * 255).astype(np.uint8)

    f = BytesIO()
    pil_image = PILImage.fromarray(image_buf_o, "RGB" if is_color else "L")

    pil_image.save(f, fmt)

    if use_widget:
        return widgets.Image(value=f.getvalue(), format=fmt)
    else:
        return Image(data=f.getvalue())


def plot_kernel(kernel, ax):
    """
    Plot a 2D kernel as a 3D surface.
    Requires an axis with projection="3d".
    """
    kernel_width = kernel.shape[0]
    xi = np.arange(kernel_width)
    x, y = np.meshgrid(xi, xi)
    ax.plot_surface(x, y, kernel, cmap="viridis")
    return ax


def plot_matrix(ax, mat, text=True, title=None, fontsize=6):
    if title:
        ax.set_title(title)

    # hide gridlines
    ax.grid(False)

    art = ax.matshow(mat)
    if np.issubdtype(mat.dtype, np.floating):
        mat = np.round(mat, 1)
    height, width = mat.shape
    if height <= 64 and width <= 64 and text:
        for i in range(height):
            for j in range(width):
                c = mat[j, i]

                # Get the color of the tile
                tile_color = art.cmap(art.norm(c))

                # Get the luminance of the color
                luminance = (
                    0.299 * tile_color[0]
                    + 0.587 * tile_color[1]
                    + 0.114 * tile_color[2]
                )
                if luminance > 0.5:
                    color = "black"
                else:
                    color = "white"

                ax.text(
                    i,
                    j,
                    str(c),
                    va="center",
                    ha="center",
                    fontsize=fontsize,
                    color=color,
                )
