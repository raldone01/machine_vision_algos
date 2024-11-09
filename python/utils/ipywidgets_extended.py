import asyncio

import ipywidgets as wid
from ipywidgets import Layout

widgets_styling = {
    "layout": Layout(width="auto"),
    "style": {"description_width": "initial"},
}


class MultiSelect:
    def __init__(
        self,
        all_choices: list[str],
        default_choices: list[str] = None,
        custom_widgets: list[wid.Widget] = None,
        grid_template_columns: str = "1fr 1fr",
    ):
        self._choices = all_choices
        self._on_selection_change = None
        self._custom_widgets = custom_widgets or [None] * len(all_choices)

        if len(self._custom_widgets) != len(self._choices):
            raise ValueError(
                "The number of custom widgets must match the number of choices"
            )

        self._checkboxes = [
            wid.Checkbox(description=choice, value=True, layout=Layout(width="auto"))
            for choice in all_choices
        ]

        for checkbox in self._checkboxes:
            checkbox.observe(self._on_checkbox_change, "value")

        self._select_all_button = wid.Button()
        self._configure_button()

        # Process selection state for default choices
        self.select(default_choices or [])

        # Build widgets: Checkbox + Custom widget (if any)
        children = [self._select_all_button]  # Include the button first
        for checkbox, custom_widget in zip(self._checkboxes, self._custom_widgets):
            if custom_widget:
                # Add checkbox followed by its related custom widget
                children.extend([wid.VBox([checkbox, custom_widget])])
            else:
                # Just add the checkbox
                children.append(checkbox)

        self._view = wid.GridBox(
            children=children,
            layout=wid.Layout(
                grid_template_columns=grid_template_columns,
                grid_gap="10px",
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


class RadioSelect:
    def __init__(
        self,
        all_choices: list[str],
        default_choice: str = None,
        custom_widgets: list[wid.Widget] = None,
        grid_template_columns: str = "1fr 1fr",
    ):
        self._choices = all_choices
        self._on_selection_change = None
        self._custom_widgets = custom_widgets or [None] * len(all_choices)

        if len(self._custom_widgets) != len(self._choices):
            raise ValueError(
                "The number of custom widgets must match the number of choices"
            )

        self._radio_buttons = [
            wid.RadioButtons(options=[choice], layout=Layout(width="max-content"))
            for choice in all_choices
        ]

        # Process selection state for the default choice
        self._select(default_choice)

        for radio_button in self._radio_buttons:
            radio_button.observe(
                lambda change, radio_button=radio_button: self._on_radio_button_change(
                    radio_button
                ),
                "value",
            )

        # Build widgets: RadioButtons + Custom widget (if any)
        children = []
        for radio_button, custom_widget in zip(
            self._radio_buttons, self._custom_widgets
        ):
            if custom_widget:
                # Add a radio button followed by its related custom widget
                children.extend([wid.VBox([radio_button, custom_widget])])
            else:
                # Just add the radio button
                children.append(radio_button)

        self._view = wid.GridBox(
            children=children,
            layout=wid.Layout(
                grid_template_columns=grid_template_columns,
                grid_gap="10px",
            ),
        )

    def set_on_selection_change(self, callback) -> None:
        self._on_selection_change = callback

    def get_view(self) -> wid.GridBox:
        return self._view

    def _on_radio_button_change(self, radio_button) -> None:
        if radio_button.index is None:
            return

        # make sure that only one radio button is selected
        for other_radio_button in self._radio_buttons:
            if other_radio_button != radio_button:
                other_radio_button.index = None

        if self._on_selection_change:
            self._on_selection_change()

    def _select(self, choice: str) -> None:
        for radio_button in self._radio_buttons:
            radio_button.index = 0 if radio_button.options[0] == choice else None

    def select(self, choice: str) -> None:
        self._select(choice)
        if self._on_selection_change:
            self._on_selection_change()

    def get_selected(self) -> str:
        for radio_button in self._radio_buttons:
            if radio_button.index == 0:
                return radio_button.options[0]


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
