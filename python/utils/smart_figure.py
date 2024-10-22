import matplotlib.pyplot as plt

open_cnt = 0


class SmartFigure:
    """
    Raii wrapper for matplotlib figures.
    It closes the figure when the object is destroyed.
    """

    def __init__(self, nrows, ncols, figsize=(8, 6), rcParams=None, **kwargs):
        with plt.ioff():
            if rcParams is not None:
                with plt.rc_context(rcParams):
                    self.fig, self.ax = plt.subplots(
                        nrows, ncols, figsize=figsize, **kwargs
                    )
            else:
                self.fig, self.ax = plt.subplots(
                    nrows, ncols, figsize=figsize, **kwargs
                )

        # One cannot define the style for minor ticks in the
        # style-sheet
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
        # print("Opening figure", self.fig, self.open_cnt)

    def __call__(self):
        return self.fig, self.ax

    def __del__(self):
        # print("Closing figure", self.fig, self.open_cnt)
        plt.close(self.fig)


def print_open_plot_count():
    open_plots_count = len(plt.get_fignums())
    print(f"There are {open_plots_count} open plots.")
