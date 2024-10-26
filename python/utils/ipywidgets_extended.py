import asyncio

import ipywidgets as wid
from ipywidgets import Layout

widgets_styling = {
    "layout": Layout(width="auto"),
    "style": {"description_width": "initial"},
}


class ManyBoxes:
    def __init__(self, all_choices: list[str], default_choices: list[str] = None):
        self._choices = all_choices
        self._on_selection_change = None

        self._checkboxes = [
            wid.Checkbox(description=choice, value=True, layout=Layout(width="auto"))
            for choice in all_choices
        ]
        for checkbox in self._checkboxes:
            checkbox.observe(self._on_checkbox_change, "value")

        self._select_all_button = wid.Button()
        self._configure_button()

        self.select(default_choices or [])

        self._view = wid.GridBox(
            children=[self._select_all_button, *self._checkboxes],
            layout=wid.Layout(
                grid_template_columns="1fr 1fr 1fr 1fr",
            ),
        )

    def set_on_selection_change(self, callback) -> None:
        self._on_selection_change = callback

    def get_view(self) -> wid.GridBox:
        return self._view

    def are_all_checked(self) -> bool:
        return all(checkbox.value for checkbox in self._checkboxes)

    def _configure_button(self) -> None:
        self._select_all_button.on_click(self._on_select_all_button_clicked)
        if self.are_all_checked():
            self._select_all_button.description = "Deselect all"
        else:
            self._select_all_button.description = "Select all"

    def _on_select_all_button_clicked(self, _) -> None:
        new_value = not self.are_all_checked()
        self.select_all(new_value)
        self._configure_button()

    def _on_checkbox_change(self, _) -> None:
        self._configure_button()

        if self._on_selection_change:
            self._on_selection_change()

    def select_all(self, value: bool) -> None:
        for checkbox in self._checkboxes:
            checkbox.value = value

    def select(self, choices: list[str]) -> None:
        for checkbox in self._checkboxes:
            checkbox.value = checkbox.description in choices

    def get_selected(self) -> list[str]:
        return [checkbox.description for checkbox in self._checkboxes if checkbox.value]


class CenteredColumn:
    def __init__(self, child):
        self._view = wid.GridBox(
            children=[wid.VBox(), child, wid.VBox()],
            layout=wid.Layout(
                grid_template_columns="1fr 3fr 1fr",
            ),
        )

    def get_view(self) -> wid.GridBox:
        return self._view


def async_observe(widget, value):
    future = asyncio.Future()

    def wrapped_callback(change):
        future.set_result(change.new)
        widget.unobserve(wrapped_callback, value)

    widget.observe(wrapped_callback, value)
    return future


def async_on_click(widget):
    future = asyncio.Future()

    def wrapped_callback(change):
        future.set_result(change)
        widget._click_handlers.callbacks = []

    widget.on_click(wrapped_callback)
    return future
