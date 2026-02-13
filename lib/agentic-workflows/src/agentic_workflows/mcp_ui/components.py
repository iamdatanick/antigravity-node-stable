"""MCP-UI Component Library.

This module provides built-in HTML templates and component factories
for common UI widgets compatible with MCP-UI.

Components include:
- Button: Interactive button component
- Input: Text input field
- Select: Dropdown selection
- Table: Data table with pagination
- Chart: Data visualization charts
- List: List display component
- Card: Card layout component

Each component generates HTML compatible with MCP-UI rendering
and can optionally generate Remote DOM scripts.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from .resource import (
    RawHtmlResource,
    wrap_html_with_adapters,
)
from .ui_types import (
    UIResource,
)


class ButtonVariant(str, Enum):
    """Button visual variants."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    DANGER = "danger"
    SUCCESS = "success"
    OUTLINE = "outline"
    GHOST = "ghost"


class ButtonSize(str, Enum):
    """Button size options."""

    SMALL = "sm"
    MEDIUM = "md"
    LARGE = "lg"


class InputType(str, Enum):
    """Input field types."""

    TEXT = "text"
    EMAIL = "email"
    PASSWORD = "password"
    NUMBER = "number"
    SEARCH = "search"
    TEL = "tel"
    URL = "url"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime-local"


@dataclass
class ButtonConfig:
    """Configuration for Button component.

    Attributes:
        label: Button text label.
        variant: Visual variant.
        size: Button size.
        disabled: Whether button is disabled.
        icon: Optional icon name.
        action: Action configuration (tool, prompt, intent, link).
        loading: Show loading state.
    """

    label: str
    variant: ButtonVariant = ButtonVariant.PRIMARY
    size: ButtonSize = ButtonSize.MEDIUM
    disabled: bool = False
    icon: str = ""
    action: dict[str, Any] | None = None
    loading: bool = False


@dataclass
class InputConfig:
    """Configuration for Input component.

    Attributes:
        name: Input field name.
        type: Input type.
        label: Display label.
        placeholder: Placeholder text.
        value: Initial value.
        required: Whether field is required.
        disabled: Whether field is disabled.
        error: Error message to display.
        help_text: Help text below input.
    """

    name: str
    type: InputType = InputType.TEXT
    label: str = ""
    placeholder: str = ""
    value: str = ""
    required: bool = False
    disabled: bool = False
    error: str = ""
    help_text: str = ""


@dataclass
class SelectOption:
    """Option for Select component.

    Attributes:
        value: Option value.
        label: Display label.
        disabled: Whether option is disabled.
    """

    value: str
    label: str
    disabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "label": self.label,
            "disabled": self.disabled,
        }


@dataclass
class SelectConfig:
    """Configuration for Select component.

    Attributes:
        name: Select field name.
        options: Available options.
        label: Display label.
        value: Selected value.
        placeholder: Placeholder text.
        required: Whether field is required.
        disabled: Whether field is disabled.
        multiple: Allow multiple selection.
    """

    name: str
    options: list[SelectOption] = field(default_factory=list)
    label: str = ""
    value: str | list[str] = ""
    placeholder: str = "Select..."
    required: bool = False
    disabled: bool = False
    multiple: bool = False


@dataclass
class TableColumn:
    """Column definition for Table component.

    Attributes:
        key: Data key for column.
        header: Column header text.
        width: Column width.
        sortable: Whether column is sortable.
        render: Custom render function name.
    """

    key: str
    header: str
    width: str = ""
    sortable: bool = False
    render: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "header": self.header,
            "width": self.width,
            "sortable": self.sortable,
            "render": self.render,
        }


@dataclass
class TableConfig:
    """Configuration for Table component.

    Attributes:
        columns: Column definitions.
        data: Table row data.
        title: Table title.
        selectable: Enable row selection.
        pagination: Enable pagination.
        page_size: Rows per page.
        loading: Show loading state.
        empty_message: Message when table is empty.
    """

    columns: list[TableColumn]
    data: list[dict[str, Any]] = field(default_factory=list)
    title: str = ""
    selectable: bool = False
    pagination: bool = True
    page_size: int = 10
    loading: bool = False
    empty_message: str = "No data available"


@dataclass
class ChartDataset:
    """Dataset for Chart component.

    Attributes:
        label: Dataset label.
        data: Data points.
        color: Line/bar color.
        background_color: Fill color.
    """

    label: str
    data: list[float | int]
    color: str = ""
    background_color: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "label": self.label,
            "data": self.data,
        }
        if self.color:
            result["borderColor"] = self.color
        if self.background_color:
            result["backgroundColor"] = self.background_color
        return result


@dataclass
class ChartConfig:
    """Configuration for Chart component.

    Attributes:
        type: Chart type (line, bar, pie, etc.).
        datasets: Chart datasets.
        labels: X-axis labels.
        title: Chart title.
        height: Chart height in pixels.
        show_legend: Show chart legend.
    """

    type: Literal["line", "bar", "pie", "doughnut", "area", "scatter"] = "line"
    datasets: list[ChartDataset] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    title: str = ""
    height: int = 300
    show_legend: bool = True


@dataclass
class ListItem:
    """Item for List component.

    Attributes:
        id: Unique item identifier.
        title: Item title.
        subtitle: Item subtitle.
        image: Image URL.
        icon: Icon name.
        badge: Badge text.
        action: Action on click.
    """

    id: str
    title: str
    subtitle: str = ""
    image: str = ""
    icon: str = ""
    badge: str = ""
    action: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "subtitle": self.subtitle,
            "image": self.image,
            "icon": self.icon,
            "badge": self.badge,
            "action": self.action,
        }


@dataclass
class ListConfig:
    """Configuration for List component.

    Attributes:
        items: List items.
        title: List title.
        selectable: Enable item selection.
        empty_message: Message when list is empty.
        loading: Show loading state.
    """

    items: list[ListItem] = field(default_factory=list)
    title: str = ""
    selectable: bool = False
    empty_message: str = "No items"
    loading: bool = False


@dataclass
class CardConfig:
    """Configuration for Card component.

    Attributes:
        title: Card title.
        subtitle: Card subtitle.
        content: Card content (HTML or text).
        image: Header image URL.
        footer: Footer content.
        actions: Card action buttons.
    """

    title: str = ""
    subtitle: str = ""
    content: str = ""
    image: str = ""
    footer: str = ""
    actions: list[ButtonConfig] = field(default_factory=list)


# Base CSS for all components
BASE_CSS = """
:root {
  --bg-color: #ffffff;
  --text-color: #1a1a1a;
  --text-muted: #6b7280;
  --border-color: #e5e7eb;
  --primary-color: #3b82f6;
  --primary-hover: #2563eb;
  --danger-color: #ef4444;
  --success-color: #10b981;
  --warning-color: #f59e0b;
  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
[data-theme="dark"] {
  --bg-color: #1f2937;
  --text-color: #f9fafb;
  --text-muted: #9ca3af;
  --border-color: #374151;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg-color);
  color: var(--text-color);
  font-size: 14px;
  line-height: 1.5;
}
"""


class Component:
    """Base class for UI components."""

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )

    @staticmethod
    def _json_attr(data: Any) -> str:
        """Convert data to JSON attribute value."""
        return json.dumps(data).replace('"', "&quot;")


class Button(Component):
    """Button component factory."""

    @staticmethod
    def render(config: ButtonConfig) -> str:
        """Render button HTML.

        Args:
            config: Button configuration.

        Returns:
            HTML string for the button.
        """
        variant_class = f"btn-{config.variant.value}"
        size_class = f"btn-{config.size.value}"
        disabled_attr = "disabled" if config.disabled else ""
        loading_class = "loading" if config.loading else ""

        action_attr = ""
        if config.action:
            action_attr = f'data-action="{Component._json_attr(config.action)}"'

        icon_html = ""
        if config.icon:
            icon_html = f'<span class="btn-icon">{config.icon}</span>'

        return f"""
<button class="btn {variant_class} {size_class} {loading_class}" {disabled_attr} {action_attr}>
  {icon_html}
  <span class="btn-label">{Component._escape_html(config.label)}</span>
</button>
"""

    @staticmethod
    def get_css() -> str:
        """Get CSS for button component."""
        return """
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 8px 16px;
  border: none;
  border-radius: var(--radius-md);
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s ease;
}
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-primary { background: var(--primary-color); color: white; }
.btn-primary:hover:not(:disabled) { background: var(--primary-hover); }
.btn-secondary { background: var(--border-color); color: var(--text-color); }
.btn-danger { background: var(--danger-color); color: white; }
.btn-success { background: var(--success-color); color: white; }
.btn-outline { background: transparent; border: 1px solid var(--border-color); color: var(--text-color); }
.btn-ghost { background: transparent; color: var(--text-color); }
.btn-sm { padding: 4px 12px; font-size: 12px; }
.btn-lg { padding: 12px 24px; font-size: 16px; }
.btn.loading .btn-label { visibility: hidden; }
.btn.loading::after {
  content: '';
  position: absolute;
  width: 16px;
  height: 16px;
  border: 2px solid currentColor;
  border-right-color: transparent;
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
"""


class Input(Component):
    """Input component factory."""

    @staticmethod
    def render(config: InputConfig) -> str:
        """Render input HTML.

        Args:
            config: Input configuration.

        Returns:
            HTML string for the input.
        """
        required_attr = "required" if config.required else ""
        disabled_attr = "disabled" if config.disabled else ""
        error_class = "error" if config.error else ""

        label_html = ""
        if config.label:
            req_mark = '<span class="required">*</span>' if config.required else ""
            label_html = f'<label class="input-label">{Component._escape_html(config.label)}{req_mark}</label>'

        error_html = ""
        if config.error:
            error_html = f'<span class="input-error">{Component._escape_html(config.error)}</span>'

        help_html = ""
        if config.help_text:
            help_html = (
                f'<span class="input-help">{Component._escape_html(config.help_text)}</span>'
            )

        return f"""
<div class="input-field">
  {label_html}
  <input
    type="{config.type.value}"
    name="{config.name}"
    class="input {error_class}"
    placeholder="{Component._escape_html(config.placeholder)}"
    value="{Component._escape_html(config.value)}"
    {required_attr}
    {disabled_attr}
  />
  {error_html}
  {help_html}
</div>
"""

    @staticmethod
    def get_css() -> str:
        """Get CSS for input component."""
        return """
.input-field { margin-bottom: 16px; }
.input-label { display: block; font-weight: 500; margin-bottom: 6px; }
.input-label .required { color: var(--danger-color); margin-left: 2px; }
.input {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  font-size: 14px;
  background: var(--bg-color);
  color: var(--text-color);
  transition: border-color 0.15s ease;
}
.input:focus { outline: none; border-color: var(--primary-color); }
.input.error { border-color: var(--danger-color); }
.input:disabled { opacity: 0.5; cursor: not-allowed; }
.input-error { display: block; color: var(--danger-color); font-size: 12px; margin-top: 4px; }
.input-help { display: block; color: var(--text-muted); font-size: 12px; margin-top: 4px; }
"""


class Select(Component):
    """Select component factory."""

    @staticmethod
    def render(config: SelectConfig) -> str:
        """Render select HTML.

        Args:
            config: Select configuration.

        Returns:
            HTML string for the select.
        """
        required_attr = "required" if config.required else ""
        disabled_attr = "disabled" if config.disabled else ""
        multiple_attr = "multiple" if config.multiple else ""

        label_html = ""
        if config.label:
            req_mark = '<span class="required">*</span>' if config.required else ""
            label_html = f'<label class="select-label">{Component._escape_html(config.label)}{req_mark}</label>'

        options_html = f'<option value="">{Component._escape_html(config.placeholder)}</option>'
        selected_values = config.value if isinstance(config.value, list) else [config.value]

        for opt in config.options:
            selected = "selected" if opt.value in selected_values else ""
            opt_disabled = "disabled" if opt.disabled else ""
            options_html += f'<option value="{opt.value}" {selected} {opt_disabled}>{Component._escape_html(opt.label)}</option>'

        return f"""
<div class="select-field">
  {label_html}
  <select name="{config.name}" class="select" {required_attr} {disabled_attr} {multiple_attr}>
    {options_html}
  </select>
</div>
"""

    @staticmethod
    def get_css() -> str:
        """Get CSS for select component."""
        return """
.select-field { margin-bottom: 16px; }
.select-label { display: block; font-weight: 500; margin-bottom: 6px; }
.select-label .required { color: var(--danger-color); margin-left: 2px; }
.select {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  font-size: 14px;
  background: var(--bg-color);
  color: var(--text-color);
  cursor: pointer;
}
.select:focus { outline: none; border-color: var(--primary-color); }
.select:disabled { opacity: 0.5; cursor: not-allowed; }
"""


class Table(Component):
    """Table component factory."""

    @staticmethod
    def render(config: TableConfig) -> str:
        """Render table HTML.

        Args:
            config: Table configuration.

        Returns:
            HTML string for the table.
        """
        if config.loading:
            return '<div class="table-loading"><div class="spinner"></div></div>'

        if not config.data:
            return f'<div class="table-empty">{Component._escape_html(config.empty_message)}</div>'

        # Title
        title_html = ""
        if config.title:
            title_html = f'<div class="table-title">{Component._escape_html(config.title)}</div>'

        # Header
        header_html = "<tr>"
        if config.selectable:
            header_html += (
                '<th class="table-select"><input type="checkbox" class="select-all" /></th>'
            )
        for col in config.columns:
            width_style = f'style="width: {col.width}"' if col.width else ""
            sortable_class = "sortable" if col.sortable else ""
            header_html += f'<th class="{sortable_class}" {width_style}>{Component._escape_html(col.header)}</th>'
        header_html += "</tr>"

        # Rows
        rows_html = ""
        for row in config.data:
            row_id = row.get("id", str(id(row)))
            rows_html += f'<tr data-id="{row_id}">'
            if config.selectable:
                rows_html += (
                    f'<td class="table-select"><input type="checkbox" value="{row_id}" /></td>'
                )
            for col in config.columns:
                value = row.get(col.key, "")
                rows_html += f"<td>{Component._escape_html(str(value))}</td>"
            rows_html += "</tr>"

        return f"""
<div class="table-container">
  {title_html}
  <table class="table">
    <thead>{header_html}</thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
"""

    @staticmethod
    def get_css() -> str:
        """Get CSS for table component."""
        return """
.table-container { overflow-x: auto; }
.table-title { font-size: 16px; font-weight: 600; margin-bottom: 12px; }
.table { width: 100%; border-collapse: collapse; }
.table th, .table td {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid var(--border-color);
}
.table th {
  font-weight: 500;
  background: var(--border-color);
}
.table th.sortable { cursor: pointer; }
.table th.sortable:hover { background: var(--text-muted); }
.table tr:hover { background: rgba(0, 0, 0, 0.02); }
.table-select { width: 40px; text-align: center; }
.table-loading, .table-empty {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 40px;
  color: var(--text-muted);
}
.spinner {
  width: 24px;
  height: 24px;
  border: 2px solid var(--border-color);
  border-top-color: var(--primary-color);
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
}
"""


class Chart(Component):
    """Chart component factory."""

    @staticmethod
    def render(config: ChartConfig) -> str:
        """Render chart HTML.

        Args:
            config: Chart configuration.

        Returns:
            HTML string for the chart.
        """
        chart_data = {
            "type": config.type,
            "data": {
                "labels": config.labels,
                "datasets": [ds.to_dict() for ds in config.datasets],
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": True,
                "plugins": {
                    "legend": {"display": config.show_legend},
                    "title": {"display": bool(config.title), "text": config.title},
                },
            },
        }

        title_html = ""
        if config.title:
            title_html = f'<div class="chart-title">{Component._escape_html(config.title)}</div>'

        return f"""
<div class="chart-container" style="height: {config.height}px">
  {title_html}
  <canvas id="chart" data-config="{Component._json_attr(chart_data)}"></canvas>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
  const config = JSON.parse(document.getElementById('chart').dataset.config);
  new Chart(document.getElementById('chart').getContext('2d'), config);
</script>
"""

    @staticmethod
    def get_css() -> str:
        """Get CSS for chart component."""
        return """
.chart-container { position: relative; width: 100%; }
.chart-title { font-size: 16px; font-weight: 600; text-align: center; margin-bottom: 12px; }
"""


class List(Component):
    """List component factory."""

    @staticmethod
    def render(config: ListConfig) -> str:
        """Render list HTML.

        Args:
            config: List configuration.

        Returns:
            HTML string for the list.
        """
        if config.loading:
            return '<div class="list-loading"><div class="spinner"></div></div>'

        if not config.items:
            return f'<div class="list-empty">{Component._escape_html(config.empty_message)}</div>'

        title_html = ""
        if config.title:
            title_html = f'<div class="list-title">{Component._escape_html(config.title)}</div>'

        items_html = ""
        for item in config.items:
            action_attr = ""
            if item.action:
                action_attr = f'data-action="{Component._json_attr(item.action)}"'

            image_html = ""
            if item.image:
                image_html = f'<img class="list-item-image" src="{item.image}" alt="" />'
            elif item.icon:
                image_html = f'<span class="list-item-icon">{item.icon}</span>'

            badge_html = ""
            if item.badge:
                badge_html = (
                    f'<span class="list-item-badge">{Component._escape_html(item.badge)}</span>'
                )

            items_html += f"""
<div class="list-item" data-id="{item.id}" {action_attr}>
  {image_html}
  <div class="list-item-content">
    <div class="list-item-title">{Component._escape_html(item.title)}</div>
    <div class="list-item-subtitle">{Component._escape_html(item.subtitle)}</div>
  </div>
  {badge_html}
</div>
"""

        return f"""
<div class="list-container">
  {title_html}
  <div class="list-items">{items_html}</div>
</div>
"""

    @staticmethod
    def get_css() -> str:
        """Get CSS for list component."""
        return """
.list-container { }
.list-title { font-size: 16px; font-weight: 600; margin-bottom: 12px; }
.list-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  border-bottom: 1px solid var(--border-color);
  cursor: pointer;
  transition: background 0.15s ease;
}
.list-item:hover { background: rgba(0, 0, 0, 0.02); }
.list-item-image {
  width: 48px;
  height: 48px;
  border-radius: var(--radius-md);
  object-fit: cover;
}
.list-item-icon {
  width: 48px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--border-color);
  border-radius: var(--radius-md);
  font-size: 20px;
}
.list-item-content { flex: 1; }
.list-item-title { font-weight: 500; }
.list-item-subtitle { font-size: 13px; color: var(--text-muted); }
.list-item-badge {
  padding: 2px 8px;
  background: var(--primary-color);
  color: white;
  border-radius: 12px;
  font-size: 12px;
}
.list-loading, .list-empty {
  display: flex;
  justify-content: center;
  padding: 40px;
  color: var(--text-muted);
}
"""


class Card(Component):
    """Card component factory."""

    @staticmethod
    def render(config: CardConfig) -> str:
        """Render card HTML.

        Args:
            config: Card configuration.

        Returns:
            HTML string for the card.
        """
        image_html = ""
        if config.image:
            image_html = f'<img class="card-image" src="{config.image}" alt="" />'

        header_html = ""
        if config.title or config.subtitle:
            subtitle_html = (
                f'<div class="card-subtitle">{Component._escape_html(config.subtitle)}</div>'
                if config.subtitle
                else ""
            )
            header_html = f"""
<div class="card-header">
  <div class="card-title">{Component._escape_html(config.title)}</div>
  {subtitle_html}
</div>
"""

        content_html = ""
        if config.content:
            content_html = f'<div class="card-content">{config.content}</div>'

        footer_html = ""
        if config.footer or config.actions:
            actions_html = ""
            for btn_config in config.actions:
                actions_html += Button.render(btn_config)
            footer_content = config.footer or actions_html
            footer_html = f'<div class="card-footer">{footer_content}</div>'

        return f"""
<div class="card">
  {image_html}
  {header_html}
  {content_html}
  {footer_html}
</div>
"""

    @staticmethod
    def get_css() -> str:
        """Get CSS for card component."""
        return """
.card {
  background: var(--bg-color);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  overflow: hidden;
  box-shadow: var(--shadow-sm);
}
.card-image {
  width: 100%;
  height: 200px;
  object-fit: cover;
}
.card-header { padding: 16px; }
.card-title { font-size: 18px; font-weight: 600; }
.card-subtitle { font-size: 14px; color: var(--text-muted); margin-top: 4px; }
.card-content { padding: 0 16px 16px; }
.card-footer {
  padding: 16px;
  border-top: 1px solid var(--border-color);
  display: flex;
  gap: 8px;
}
"""


def create_component_html(
    component_type: Literal["button", "input", "select", "table", "chart", "list", "card"],
    config: Any,
    include_base_css: bool = True,
    enable_apps_sdk: bool = False,
) -> str:
    """Create complete HTML for a component.

    Args:
        component_type: Type of component to render.
        config: Component configuration.
        include_base_css: Include base CSS in output.
        enable_apps_sdk: Enable Apps SDK adapter.

    Returns:
        Complete HTML string with styles.
    """
    component_map = {
        "button": (Button, ButtonConfig),
        "input": (Input, InputConfig),
        "select": (Select, SelectConfig),
        "table": (Table, TableConfig),
        "chart": (Chart, ChartConfig),
        "list": (List, ListConfig),
        "card": (Card, CardConfig),
    }

    if component_type not in component_map:
        raise ValueError(f"Unknown component type: {component_type}")

    component_class, _ = component_map[component_type]
    component_html = component_class.render(config)
    component_css = component_class.get_css()

    css = ""
    if include_base_css:
        css = BASE_CSS
    css += component_css

    html = f"""
<!DOCTYPE html>
<html>
<head>
<style>{css}</style>
</head>
<body>
{component_html}
<script>
document.body.dataset.theme = window.mcpui?.getTheme() || window.openai?.theme || 'light';
document.querySelectorAll('[data-action]').forEach(el => {{
  el.addEventListener('click', () => {{
    const action = JSON.parse(el.dataset.action);
    if (window.mcpui) {{
      if (action.type === 'tool') window.mcpui.callTool(action.toolName, action.params);
      else if (action.type === 'prompt') window.mcpui.sendPrompt(action.prompt);
      else if (action.type === 'link') window.mcpui.openLink(action.url);
    }}
  }});
}});
</script>
</body>
</html>
"""

    if enable_apps_sdk:
        html = wrap_html_with_adapters(html, {"appsSdk": {"enabled": True}})

    return html


def create_component_resource(
    component_type: Literal["button", "input", "select", "table", "chart", "list", "card"],
    config: Any,
    uri: str | None = None,
    enable_apps_sdk: bool = True,
) -> UIResource:
    """Create a UIResource for a component.

    Args:
        component_type: Type of component.
        config: Component configuration.
        uri: Optional resource URI.
        enable_apps_sdk: Enable Apps SDK adapter.

    Returns:
        UIResource with component HTML.
    """
    html = create_component_html(
        component_type,
        config,
        include_base_css=True,
        enable_apps_sdk=enable_apps_sdk,
    )

    if not uri:
        uri = f"ui://{component_type}/{uuid.uuid4().hex[:8]}"

    return RawHtmlResource.create(
        html=html,
        uri=uri,
        enable_apps_sdk=enable_apps_sdk,
    )


def generate_remote_dom_script(
    component_type: str,
    config: Any,
) -> str:
    """Generate Remote DOM script for a component.

    Creates a JavaScript script that can be executed in the
    Remote DOM environment to render the component.

    Args:
        component_type: Type of component.
        config: Component configuration.

    Returns:
        JavaScript code for Remote DOM execution.
    """
    config_json = json.dumps(config.__dict__ if hasattr(config, "__dict__") else config)

    return f"""
// Remote DOM Component: {component_type}
(function() {{
  const config = {config_json};
  const root = document.getElementById('root') || document.body;

  // Apply theme from host
  const theme = window.__MCP_UI_THEME__ || 'light';
  document.body.dataset.theme = theme;

  // Render component
  const componentHtml = window.__MCP_UI_COMPONENTS__?.['{component_type}']?.(config) || '';
  root.innerHTML = componentHtml;

  // Setup action handlers
  document.querySelectorAll('[data-action]').forEach(el => {{
    el.addEventListener('click', () => {{
      const action = JSON.parse(el.dataset.action);
      window.parent.postMessage({{
        type: 'mcp-ui:action',
        action: action
      }}, '*');
    }});
  }});
}})();
"""
