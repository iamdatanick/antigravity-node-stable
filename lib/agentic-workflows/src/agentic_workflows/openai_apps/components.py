"""Widget components for OpenAI Apps SDK integration.

This module provides pre-built widget components:
- ListViewWidget: Display lists of items with selection
- CarouselWidget: Image/card carousels
- FormWidget: Interactive forms with inputs
- StatusWidget: Progress/status displays
- ChartWidget: Data visualization charts

Each component includes HTML templates and payload classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .widget_types import (
    WidgetPayload,
    WidgetTemplate,
    WidgetMeta,
    StructuredContent,
)


class ListStyle(str, Enum):
    """List display styles."""

    DEFAULT = "default"
    GRID = "grid"
    COMPACT = "compact"
    DETAILED = "detailed"


@dataclass
class ListItem:
    """A single item in a list view.

    Attributes:
        id: Unique item identifier
        title: Item title
        subtitle: Optional subtitle
        description: Optional description
        image_url: Optional image URL
        icon: Optional icon name
        badge: Optional badge text
        metadata: Additional item metadata
        actions: Available actions for this item
        selected: Whether item is selected
    """

    id: str
    title: str
    subtitle: str = ""
    description: str = ""
    image_url: str = ""
    icon: str = ""
    badge: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    actions: list[dict[str, str]] = field(default_factory=list)
    selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "subtitle": self.subtitle,
            "description": self.description,
            "imageUrl": self.image_url,
            "icon": self.icon,
            "badge": self.badge,
            "metadata": self.metadata,
            "actions": self.actions,
            "selected": self.selected,
        }


class ListViewPayload(WidgetPayload[list[ListItem]]):
    """Payload for list view widgets.

    Example:
        >>> items = [
        ...     ListItem(id="1", title="Product A", subtitle="$29.99"),
        ...     ListItem(id="2", title="Product B", subtitle="$49.99"),
        ... ]
        >>> payload = ListViewPayload(items=items, title="Products")
        >>> content = payload.with_meta(WidgetMeta(output_template="widget://list-view"))
    """

    def __init__(
        self,
        items: list[ListItem],
        title: str = "",
        style: ListStyle = ListStyle.DEFAULT,
        selectable: bool = False,
        multi_select: bool = False,
        empty_message: str = "No items to display",
        loading: bool = False,
        total_count: int | None = None,
        page: int = 1,
        page_size: int = 20,
    ):
        """Initialize list view payload.

        Args:
            items: List of items to display.
            title: Optional list title.
            style: List display style.
            selectable: Enable item selection.
            multi_select: Allow multiple selection.
            empty_message: Message when list is empty.
            loading: Show loading indicator.
            total_count: Total items for pagination.
            page: Current page number.
            page_size: Items per page.
        """
        self.items = items
        self.title = title
        self.style = style
        self.selectable = selectable
        self.multi_select = multi_select
        self.empty_message = empty_message
        self.loading = loading
        self.total_count = total_count if total_count is not None else len(items)
        self.page = page
        self.page_size = page_size

    def to_structured_content(self) -> StructuredContent:
        """Convert to structured content."""
        return {
            "type": "list_view",
            "title": self.title,
            "items": [item.to_dict() for item in self.items],
            "style": self.style.value,
            "selectable": self.selectable,
            "multiSelect": self.multi_select,
            "emptyMessage": self.empty_message,
            "loading": self.loading,
            "pagination": {
                "totalCount": self.total_count,
                "page": self.page,
                "pageSize": self.page_size,
                "totalPages": (self.total_count + self.page_size - 1) // self.page_size,
            },
        }


class CarouselItemType(str, Enum):
    """Carousel item types."""

    IMAGE = "image"
    CARD = "card"
    VIDEO = "video"


@dataclass
class CarouselItem:
    """A single item in a carousel.

    Attributes:
        id: Unique item identifier
        type: Item type (image, card, video)
        title: Item title
        description: Optional description
        image_url: Image URL
        video_url: Video URL (for video type)
        link_url: Optional click-through URL
        metadata: Additional item metadata
    """

    id: str
    type: CarouselItemType = CarouselItemType.IMAGE
    title: str = ""
    description: str = ""
    image_url: str = ""
    video_url: str = ""
    link_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "imageUrl": self.image_url,
            "videoUrl": self.video_url,
            "linkUrl": self.link_url,
            "metadata": self.metadata,
        }


class CarouselPayload(WidgetPayload[list[CarouselItem]]):
    """Payload for carousel widgets.

    Example:
        >>> items = [
        ...     CarouselItem(id="1", image_url="https://example.com/img1.jpg", title="Image 1"),
        ...     CarouselItem(id="2", image_url="https://example.com/img2.jpg", title="Image 2"),
        ... ]
        >>> payload = CarouselPayload(items=items, auto_play=True)
    """

    def __init__(
        self,
        items: list[CarouselItem],
        title: str = "",
        auto_play: bool = False,
        auto_play_interval: int = 5000,
        show_arrows: bool = True,
        show_dots: bool = True,
        loop: bool = True,
        items_per_view: int = 1,
    ):
        """Initialize carousel payload.

        Args:
            items: List of carousel items.
            title: Optional carousel title.
            auto_play: Enable auto-play.
            auto_play_interval: Auto-play interval in ms.
            show_arrows: Show navigation arrows.
            show_dots: Show pagination dots.
            loop: Enable infinite loop.
            items_per_view: Items visible at once.
        """
        self.items = items
        self.title = title
        self.auto_play = auto_play
        self.auto_play_interval = auto_play_interval
        self.show_arrows = show_arrows
        self.show_dots = show_dots
        self.loop = loop
        self.items_per_view = items_per_view

    def to_structured_content(self) -> StructuredContent:
        """Convert to structured content."""
        return {
            "type": "carousel",
            "title": self.title,
            "items": [item.to_dict() for item in self.items],
            "autoPlay": self.auto_play,
            "autoPlayInterval": self.auto_play_interval,
            "showArrows": self.show_arrows,
            "showDots": self.show_dots,
            "loop": self.loop,
            "itemsPerView": self.items_per_view,
        }


class FormFieldType(str, Enum):
    """Form field types."""

    TEXT = "text"
    EMAIL = "email"
    PASSWORD = "password"
    NUMBER = "number"
    TEXTAREA = "textarea"
    SELECT = "select"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    FILE = "file"
    HIDDEN = "hidden"


@dataclass
class FormField:
    """A form input field.

    Attributes:
        name: Field name (form key)
        type: Field type
        label: Display label
        placeholder: Placeholder text
        value: Current/default value
        required: Field is required
        disabled: Field is disabled
        options: Options for select/radio
        validation: Validation rules
        help_text: Help text below field
        error: Error message to display
    """

    name: str
    type: FormFieldType = FormFieldType.TEXT
    label: str = ""
    placeholder: str = ""
    value: Any = ""
    required: bool = False
    disabled: bool = False
    options: list[dict[str, str]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    help_text: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "label": self.label or self.name.replace("_", " ").title(),
            "placeholder": self.placeholder,
            "value": self.value,
            "required": self.required,
            "disabled": self.disabled,
            "options": self.options,
            "validation": self.validation,
            "helpText": self.help_text,
            "error": self.error,
        }


class FormPayload(WidgetPayload[list[FormField]]):
    """Payload for form widgets.

    Example:
        >>> fields = [
        ...     FormField(name="email", type=FormFieldType.EMAIL, required=True),
        ...     FormField(name="password", type=FormFieldType.PASSWORD, required=True),
        ... ]
        >>> payload = FormPayload(fields=fields, submit_label="Sign In")
    """

    def __init__(
        self,
        fields: list[FormField],
        title: str = "",
        description: str = "",
        submit_label: str = "Submit",
        cancel_label: str = "",
        action: str = "",
        method: str = "POST",
        loading: bool = False,
        success_message: str = "",
        error_message: str = "",
        sections: list[dict[str, Any]] | None = None,
    ):
        """Initialize form payload.

        Args:
            fields: List of form fields.
            title: Form title.
            description: Form description.
            submit_label: Submit button label.
            cancel_label: Cancel button label (empty to hide).
            action: Form action URL.
            method: Form method.
            loading: Show loading state.
            success_message: Success message to display.
            error_message: Error message to display.
            sections: Optional field groupings.
        """
        self.fields = fields
        self.title = title
        self.description = description
        self.submit_label = submit_label
        self.cancel_label = cancel_label
        self.action = action
        self.method = method
        self.loading = loading
        self.success_message = success_message
        self.error_message = error_message
        self.sections = sections or []

    def to_structured_content(self) -> StructuredContent:
        """Convert to structured content."""
        return {
            "type": "form",
            "title": self.title,
            "description": self.description,
            "fields": [field.to_dict() for field in self.fields],
            "submitLabel": self.submit_label,
            "cancelLabel": self.cancel_label,
            "action": self.action,
            "method": self.method,
            "loading": self.loading,
            "successMessage": self.success_message,
            "errorMessage": self.error_message,
            "sections": self.sections,
        }


class StatusType(str, Enum):
    """Status indicator types."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    LOADING = "loading"


@dataclass
class StatusStep:
    """A step in a multi-step status.

    Attributes:
        id: Step identifier
        title: Step title
        description: Step description
        status: Step status
        progress: Progress percentage (0-100)
        metadata: Additional step metadata
    """

    id: str
    title: str
    description: str = ""
    status: StatusType = StatusType.INFO
    progress: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "progress": self.progress,
            "metadata": self.metadata,
        }


class StatusPayload(WidgetPayload[None]):
    """Payload for status/progress widgets.

    Example:
        >>> payload = StatusPayload(
        ...     title="Processing",
        ...     status=StatusType.LOADING,
        ...     progress=45,
        ...     message="Analyzing data..."
        ... )
    """

    def __init__(
        self,
        title: str,
        status: StatusType = StatusType.INFO,
        message: str = "",
        progress: int | None = None,
        show_progress_bar: bool = True,
        steps: list[StatusStep] | None = None,
        current_step: int = 0,
        icon: str = "",
        actions: list[dict[str, str]] | None = None,
        timestamp: str = "",
        eta: str = "",
    ):
        """Initialize status payload.

        Args:
            title: Status title.
            status: Status type.
            message: Status message.
            progress: Progress percentage (0-100).
            show_progress_bar: Show progress bar.
            steps: Multi-step status.
            current_step: Current step index.
            icon: Custom icon.
            actions: Available actions.
            timestamp: Timestamp string.
            eta: Estimated time remaining.
        """
        self.title = title
        self.status = status
        self.message = message
        self.progress = progress
        self.show_progress_bar = show_progress_bar
        self.steps = steps or []
        self.current_step = current_step
        self.icon = icon
        self.actions = actions or []
        self.timestamp = timestamp
        self.eta = eta

    def to_structured_content(self) -> StructuredContent:
        """Convert to structured content."""
        content: StructuredContent = {
            "type": "status",
            "title": self.title,
            "status": self.status.value,
            "message": self.message,
            "showProgressBar": self.show_progress_bar,
            "icon": self.icon,
            "actions": self.actions,
            "timestamp": self.timestamp,
            "eta": self.eta,
        }

        if self.progress is not None:
            content["progress"] = self.progress

        if self.steps:
            content["steps"] = [step.to_dict() for step in self.steps]
            content["currentStep"] = self.current_step

        return content


class ChartType(str, Enum):
    """Chart types."""

    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    DOUGHNUT = "doughnut"
    AREA = "area"
    SCATTER = "scatter"
    RADAR = "radar"
    POLAR = "polar"
    TREEMAP = "treemap"
    HEATMAP = "heatmap"


@dataclass
class ChartDataset:
    """A dataset for a chart.

    Attributes:
        label: Dataset label
        data: Data points
        color: Line/bar color
        background_color: Fill color
        border_color: Border color
        fill: Fill area under line
        tension: Line tension (smoothing)
        metadata: Additional dataset metadata
    """

    label: str
    data: list[float | int | dict[str, Any]]
    color: str = ""
    background_color: str = ""
    border_color: str = ""
    fill: bool = False
    tension: float = 0.1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "label": self.label,
            "data": self.data,
            "fill": self.fill,
            "tension": self.tension,
            "metadata": self.metadata,
        }

        if self.color:
            result["color"] = self.color
        if self.background_color:
            result["backgroundColor"] = self.background_color
        if self.border_color:
            result["borderColor"] = self.border_color

        return result


class ChartPayload(WidgetPayload[list[ChartDataset]]):
    """Payload for chart widgets.

    Example:
        >>> datasets = [
        ...     ChartDataset(
        ...         label="Revenue",
        ...         data=[100, 200, 150, 300, 250],
        ...         color="#4CAF50"
        ...     )
        ... ]
        >>> payload = ChartPayload(
        ...     chart_type=ChartType.LINE,
        ...     labels=["Jan", "Feb", "Mar", "Apr", "May"],
        ...     datasets=datasets,
        ...     title="Monthly Revenue"
        ... )
    """

    def __init__(
        self,
        chart_type: ChartType,
        datasets: list[ChartDataset],
        labels: list[str] | None = None,
        title: str = "",
        subtitle: str = "",
        x_axis_label: str = "",
        y_axis_label: str = "",
        show_legend: bool = True,
        legend_position: str = "top",
        show_grid: bool = True,
        responsive: bool = True,
        maintain_aspect_ratio: bool = True,
        height: int | None = None,
        width: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """Initialize chart payload.

        Args:
            chart_type: Type of chart.
            datasets: Chart datasets.
            labels: X-axis labels.
            title: Chart title.
            subtitle: Chart subtitle.
            x_axis_label: X-axis label.
            y_axis_label: Y-axis label.
            show_legend: Show legend.
            legend_position: Legend position.
            show_grid: Show grid lines.
            responsive: Responsive sizing.
            maintain_aspect_ratio: Maintain aspect ratio.
            height: Fixed height in pixels.
            width: Fixed width in pixels.
            options: Additional chart.js options.
        """
        self.chart_type = chart_type
        self.datasets = datasets
        self.labels = labels or []
        self.title = title
        self.subtitle = subtitle
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.show_legend = show_legend
        self.legend_position = legend_position
        self.show_grid = show_grid
        self.responsive = responsive
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.height = height
        self.width = width
        self.options = options or {}

    def to_structured_content(self) -> StructuredContent:
        """Convert to structured content."""
        content: StructuredContent = {
            "type": "chart",
            "chartType": self.chart_type.value,
            "title": self.title,
            "subtitle": self.subtitle,
            "data": {
                "labels": self.labels,
                "datasets": [ds.to_dict() for ds in self.datasets],
            },
            "options": {
                "responsive": self.responsive,
                "maintainAspectRatio": self.maintain_aspect_ratio,
                "plugins": {
                    "legend": {
                        "display": self.show_legend,
                        "position": self.legend_position,
                    },
                    "title": {
                        "display": bool(self.title),
                        "text": self.title,
                    },
                },
                "scales": {},
                **self.options,
            },
        }

        # Add axis labels if provided
        if self.x_axis_label or self.y_axis_label or self.show_grid:
            content["options"]["scales"] = {
                "x": {
                    "display": True,
                    "grid": {"display": self.show_grid},
                    "title": {
                        "display": bool(self.x_axis_label),
                        "text": self.x_axis_label,
                    },
                },
                "y": {
                    "display": True,
                    "grid": {"display": self.show_grid},
                    "title": {
                        "display": bool(self.y_axis_label),
                        "text": self.y_axis_label,
                    },
                },
            }

        if self.height:
            content["height"] = self.height
        if self.width:
            content["width"] = self.width

        return content


# Widget Templates
# These provide the HTML/JS templates for each widget type


LIST_VIEW_TEMPLATE = WidgetTemplate(
    uri="widget://list-view",
    name="ListView",
    description="Display lists of items with optional selection",
    html="""
<!DOCTYPE html>
<html>
<head>
<style>
:root {
  --bg-color: #ffffff;
  --text-color: #1a1a1a;
  --border-color: #e5e5e5;
  --hover-bg: #f5f5f5;
  --selected-bg: #e3f2fd;
  --accent-color: #2196f3;
}
[data-theme="dark"] {
  --bg-color: #1a1a1a;
  --text-color: #ffffff;
  --border-color: #333333;
  --hover-bg: #2a2a2a;
  --selected-bg: #1e3a5f;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg-color);
  color: var(--text-color);
  padding: 16px;
}
.list-container { max-width: 600px; margin: 0 auto; }
.list-title {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 16px;
}
.list-item {
  display: flex;
  align-items: center;
  padding: 12px;
  border-bottom: 1px solid var(--border-color);
  cursor: pointer;
  transition: background 0.2s;
}
.list-item:hover { background: var(--hover-bg); }
.list-item.selected { background: var(--selected-bg); }
.list-item-image {
  width: 48px;
  height: 48px;
  border-radius: 8px;
  object-fit: cover;
  margin-right: 12px;
}
.list-item-content { flex: 1; }
.list-item-title { font-weight: 500; margin-bottom: 4px; }
.list-item-subtitle { font-size: 14px; opacity: 0.7; }
.list-item-badge {
  background: var(--accent-color);
  color: white;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 12px;
}
.empty-state {
  text-align: center;
  padding: 40px;
  opacity: 0.6;
}
.loading {
  display: flex;
  justify-content: center;
  padding: 20px;
}
.spinner {
  width: 24px;
  height: 24px;
  border: 2px solid var(--border-color);
  border-top-color: var(--accent-color);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div id="root"></div>
<script>
const { toolOutput, theme, widgetState, setWidgetState, callTool } = window.openai || {};
document.body.dataset.theme = theme || 'light';

function render(data) {
  const root = document.getElementById('root');
  if (data.loading) {
    root.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    return;
  }

  if (!data.items || data.items.length === 0) {
    root.innerHTML = '<div class="empty-state">' + (data.emptyMessage || 'No items') + '</div>';
    return;
  }

  let html = '<div class="list-container">';
  if (data.title) html += '<div class="list-title">' + data.title + '</div>';

  data.items.forEach(item => {
    const selected = widgetState?.selectedIds?.includes(item.id) ? 'selected' : '';
    html += '<div class="list-item ' + selected + '" data-id="' + item.id + '">';
    if (item.imageUrl) html += '<img class="list-item-image" src="' + item.imageUrl + '" />';
    html += '<div class="list-item-content">';
    html += '<div class="list-item-title">' + item.title + '</div>';
    if (item.subtitle) html += '<div class="list-item-subtitle">' + item.subtitle + '</div>';
    html += '</div>';
    if (item.badge) html += '<span class="list-item-badge">' + item.badge + '</span>';
    html += '</div>';
  });

  html += '</div>';
  root.innerHTML = html;

  if (data.selectable) {
    root.querySelectorAll('.list-item').forEach(el => {
      el.addEventListener('click', () => {
        const id = el.dataset.id;
        const selectedIds = widgetState?.selectedIds || [];
        const newIds = data.multiSelect
          ? selectedIds.includes(id)
            ? selectedIds.filter(i => i !== id)
            : [...selectedIds, id]
          : [id];
        setWidgetState?.({ selectedIds: newIds });
      });
    });
  }
}

if (toolOutput) render(toolOutput);
</script>
</body>
</html>
""",
)


CAROUSEL_TEMPLATE = WidgetTemplate(
    uri="widget://carousel",
    name="Carousel",
    description="Image/card carousel with navigation",
    html="""
<!DOCTYPE html>
<html>
<head>
<style>
:root {
  --bg-color: #ffffff;
  --text-color: #1a1a1a;
  --arrow-bg: rgba(0,0,0,0.5);
}
[data-theme="dark"] {
  --bg-color: #1a1a1a;
  --text-color: #ffffff;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg-color);
  color: var(--text-color);
}
.carousel {
  position: relative;
  overflow: hidden;
  max-width: 800px;
  margin: 0 auto;
}
.carousel-track {
  display: flex;
  transition: transform 0.3s ease;
}
.carousel-item {
  flex: 0 0 100%;
  padding: 16px;
}
.carousel-item img {
  width: 100%;
  height: 300px;
  object-fit: cover;
  border-radius: 12px;
}
.carousel-item-title {
  font-weight: 600;
  margin-top: 12px;
}
.carousel-item-desc {
  font-size: 14px;
  opacity: 0.7;
  margin-top: 4px;
}
.carousel-arrow {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  width: 40px;
  height: 40px;
  background: var(--arrow-bg);
  color: white;
  border: none;
  border-radius: 50%;
  cursor: pointer;
  font-size: 18px;
  z-index: 10;
}
.carousel-arrow.prev { left: 8px; }
.carousel-arrow.next { right: 8px; }
.carousel-dots {
  display: flex;
  justify-content: center;
  gap: 8px;
  padding: 16px;
}
.carousel-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--text-color);
  opacity: 0.3;
  cursor: pointer;
}
.carousel-dot.active { opacity: 1; }
</style>
</head>
<body>
<div id="root"></div>
<script>
const { toolOutput, theme, widgetState, setWidgetState } = window.openai || {};
document.body.dataset.theme = theme || 'light';

let currentIndex = widgetState?.currentIndex || 0;
let autoPlayInterval;

function render(data) {
  const root = document.getElementById('root');
  const items = data.items || [];

  let html = '<div class="carousel">';
  html += '<div class="carousel-track" style="transform: translateX(-' + (currentIndex * 100) + '%)">';

  items.forEach(item => {
    html += '<div class="carousel-item">';
    if (item.imageUrl) html += '<img src="' + item.imageUrl + '" alt="' + item.title + '" />';
    if (item.title) html += '<div class="carousel-item-title">' + item.title + '</div>';
    if (item.description) html += '<div class="carousel-item-desc">' + item.description + '</div>';
    html += '</div>';
  });

  html += '</div>';

  if (data.showArrows && items.length > 1) {
    html += '<button class="carousel-arrow prev">&lt;</button>';
    html += '<button class="carousel-arrow next">&gt;</button>';
  }

  if (data.showDots && items.length > 1) {
    html += '<div class="carousel-dots">';
    items.forEach((_, i) => {
      html += '<div class="carousel-dot' + (i === currentIndex ? ' active' : '') + '" data-index="' + i + '"></div>';
    });
    html += '</div>';
  }

  html += '</div>';
  root.innerHTML = html;

  const track = root.querySelector('.carousel-track');
  const prev = root.querySelector('.carousel-arrow.prev');
  const next = root.querySelector('.carousel-arrow.next');
  const dots = root.querySelectorAll('.carousel-dot');

  function goTo(index) {
    if (data.loop) {
      currentIndex = (index + items.length) % items.length;
    } else {
      currentIndex = Math.max(0, Math.min(index, items.length - 1));
    }
    track.style.transform = 'translateX(-' + (currentIndex * 100) + '%)';
    dots.forEach((dot, i) => dot.classList.toggle('active', i === currentIndex));
    setWidgetState?.({ currentIndex });
  }

  prev?.addEventListener('click', () => goTo(currentIndex - 1));
  next?.addEventListener('click', () => goTo(currentIndex + 1));
  dots.forEach(dot => dot.addEventListener('click', () => goTo(parseInt(dot.dataset.index))));

  if (data.autoPlay && items.length > 1) {
    autoPlayInterval = setInterval(() => goTo(currentIndex + 1), data.autoPlayInterval || 5000);
  }
}

if (toolOutput) render(toolOutput);
</script>
</body>
</html>
""",
)


FORM_TEMPLATE = WidgetTemplate(
    uri="widget://form",
    name="Form",
    description="Interactive form with validation",
    html="""
<!DOCTYPE html>
<html>
<head>
<style>
:root {
  --bg-color: #ffffff;
  --text-color: #1a1a1a;
  --border-color: #e5e5e5;
  --accent-color: #2196f3;
  --error-color: #f44336;
  --success-color: #4caf50;
}
[data-theme="dark"] {
  --bg-color: #1a1a1a;
  --text-color: #ffffff;
  --border-color: #333333;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg-color);
  color: var(--text-color);
  padding: 16px;
}
.form-container { max-width: 500px; margin: 0 auto; }
.form-title { font-size: 20px; font-weight: 600; margin-bottom: 8px; }
.form-desc { font-size: 14px; opacity: 0.7; margin-bottom: 24px; }
.form-field { margin-bottom: 16px; }
.form-label { display: block; font-weight: 500; margin-bottom: 6px; font-size: 14px; }
.form-label .required { color: var(--error-color); }
.form-input, .form-select, .form-textarea {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  font-size: 14px;
  background: var(--bg-color);
  color: var(--text-color);
}
.form-input:focus, .form-select:focus, .form-textarea:focus {
  outline: none;
  border-color: var(--accent-color);
}
.form-input.error { border-color: var(--error-color); }
.form-error { color: var(--error-color); font-size: 12px; margin-top: 4px; }
.form-help { font-size: 12px; opacity: 0.6; margin-top: 4px; }
.form-actions { display: flex; gap: 12px; margin-top: 24px; }
.form-btn {
  padding: 10px 20px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
}
.form-btn.primary { background: var(--accent-color); color: white; }
.form-btn.secondary { background: var(--border-color); color: var(--text-color); }
.form-message {
  padding: 12px;
  border-radius: 6px;
  margin-bottom: 16px;
}
.form-message.success { background: var(--success-color); color: white; }
.form-message.error { background: var(--error-color); color: white; }
</style>
</head>
<body>
<div id="root"></div>
<script>
const { toolOutput, theme, widgetState, setWidgetState, callTool } = window.openai || {};
document.body.dataset.theme = theme || 'light';

function render(data) {
  const root = document.getElementById('root');
  const formData = widgetState?.formData || {};

  let html = '<div class="form-container">';

  if (data.successMessage) {
    html += '<div class="form-message success">' + data.successMessage + '</div>';
  }
  if (data.errorMessage) {
    html += '<div class="form-message error">' + data.errorMessage + '</div>';
  }

  if (data.title) html += '<div class="form-title">' + data.title + '</div>';
  if (data.description) html += '<div class="form-desc">' + data.description + '</div>';

  html += '<form id="widget-form">';

  (data.fields || []).forEach(field => {
    html += '<div class="form-field">';
    html += '<label class="form-label">' + field.label;
    if (field.required) html += ' <span class="required">*</span>';
    html += '</label>';

    const value = formData[field.name] || field.value || '';
    const errorClass = field.error ? ' error' : '';

    if (field.type === 'textarea') {
      html += '<textarea class="form-textarea' + errorClass + '" name="' + field.name + '" placeholder="' + (field.placeholder || '') + '">' + value + '</textarea>';
    } else if (field.type === 'select') {
      html += '<select class="form-select' + errorClass + '" name="' + field.name + '">';
      html += '<option value="">Select...</option>';
      (field.options || []).forEach(opt => {
        const sel = opt.value === value ? ' selected' : '';
        html += '<option value="' + opt.value + '"' + sel + '>' + opt.label + '</option>';
      });
      html += '</select>';
    } else if (field.type === 'checkbox') {
      const checked = value ? ' checked' : '';
      html += '<input type="checkbox" name="' + field.name + '"' + checked + ' />';
    } else {
      html += '<input class="form-input' + errorClass + '" type="' + field.type + '" name="' + field.name + '" value="' + value + '" placeholder="' + (field.placeholder || '') + '" />';
    }

    if (field.error) html += '<div class="form-error">' + field.error + '</div>';
    if (field.helpText) html += '<div class="form-help">' + field.helpText + '</div>';
    html += '</div>';
  });

  html += '<div class="form-actions">';
  html += '<button type="submit" class="form-btn primary">' + (data.submitLabel || 'Submit') + '</button>';
  if (data.cancelLabel) {
    html += '<button type="button" class="form-btn secondary cancel">' + data.cancelLabel + '</button>';
  }
  html += '</div></form></div>';

  root.innerHTML = html;

  const form = document.getElementById('widget-form');
  form.addEventListener('input', (e) => {
    const fd = { ...formData };
    if (e.target.type === 'checkbox') {
      fd[e.target.name] = e.target.checked;
    } else {
      fd[e.target.name] = e.target.value;
    }
    setWidgetState?.({ formData: fd });
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fd = Object.fromEntries(new FormData(form));
    await callTool?.('form_submit', { formData: fd, action: data.action });
  });
}

if (toolOutput) render(toolOutput);
</script>
</body>
</html>
""",
)


STATUS_TEMPLATE = WidgetTemplate(
    uri="widget://status",
    name="Status",
    description="Progress and status display",
    html="""
<!DOCTYPE html>
<html>
<head>
<style>
:root {
  --bg-color: #ffffff;
  --text-color: #1a1a1a;
  --border-color: #e5e5e5;
  --info-color: #2196f3;
  --success-color: #4caf50;
  --warning-color: #ff9800;
  --error-color: #f44336;
}
[data-theme="dark"] {
  --bg-color: #1a1a1a;
  --text-color: #ffffff;
  --border-color: #333333;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg-color);
  color: var(--text-color);
  padding: 16px;
}
.status-container { max-width: 500px; margin: 0 auto; }
.status-header { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
.status-icon {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
}
.status-icon.info { background: var(--info-color); color: white; }
.status-icon.success { background: var(--success-color); color: white; }
.status-icon.warning { background: var(--warning-color); color: white; }
.status-icon.error { background: var(--error-color); color: white; }
.status-icon.loading { background: var(--border-color); }
.status-title { font-size: 18px; font-weight: 600; }
.status-message { font-size: 14px; opacity: 0.7; margin-bottom: 16px; }
.progress-container { margin-bottom: 16px; }
.progress-bar {
  height: 8px;
  background: var(--border-color);
  border-radius: 4px;
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  background: var(--info-color);
  transition: width 0.3s ease;
}
.progress-text { font-size: 12px; margin-top: 4px; text-align: right; }
.steps-container { margin-top: 20px; }
.step {
  display: flex;
  align-items: flex-start;
  padding: 12px 0;
  border-left: 2px solid var(--border-color);
  padding-left: 20px;
  position: relative;
}
.step::before {
  content: '';
  position: absolute;
  left: -6px;
  top: 14px;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--border-color);
}
.step.completed::before { background: var(--success-color); }
.step.active::before { background: var(--info-color); }
.step.error::before { background: var(--error-color); }
.step-content { flex: 1; }
.step-title { font-weight: 500; }
.step-desc { font-size: 13px; opacity: 0.7; margin-top: 2px; }
.spinner {
  width: 24px;
  height: 24px;
  border: 3px solid var(--border-color);
  border-top-color: var(--info-color);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.status-meta { display: flex; gap: 16px; font-size: 12px; opacity: 0.6; margin-top: 12px; }
</style>
</head>
<body>
<div id="root"></div>
<script>
const { toolOutput, theme } = window.openai || {};
document.body.dataset.theme = theme || 'light';

const icons = { info: 'i', success: '\\u2713', warning: '!', error: '\\u2717' };

function render(data) {
  const root = document.getElementById('root');

  let html = '<div class="status-container">';
  html += '<div class="status-header">';

  if (data.status === 'loading') {
    html += '<div class="spinner"></div>';
  } else {
    html += '<div class="status-icon ' + data.status + '">' + (data.icon || icons[data.status] || 'i') + '</div>';
  }

  html += '<div class="status-title">' + data.title + '</div>';
  html += '</div>';

  if (data.message) html += '<div class="status-message">' + data.message + '</div>';

  if (data.showProgressBar && data.progress !== undefined) {
    html += '<div class="progress-container">';
    html += '<div class="progress-bar"><div class="progress-fill" style="width: ' + data.progress + '%"></div></div>';
    html += '<div class="progress-text">' + data.progress + '%</div>';
    html += '</div>';
  }

  if (data.steps && data.steps.length > 0) {
    html += '<div class="steps-container">';
    data.steps.forEach((step, i) => {
      const stepClass = i < data.currentStep ? 'completed' : i === data.currentStep ? 'active' : '';
      html += '<div class="step ' + stepClass + '">';
      html += '<div class="step-content">';
      html += '<div class="step-title">' + step.title + '</div>';
      if (step.description) html += '<div class="step-desc">' + step.description + '</div>';
      html += '</div></div>';
    });
    html += '</div>';
  }

  if (data.timestamp || data.eta) {
    html += '<div class="status-meta">';
    if (data.timestamp) html += '<span>Updated: ' + data.timestamp + '</span>';
    if (data.eta) html += '<span>ETA: ' + data.eta + '</span>';
    html += '</div>';
  }

  html += '</div>';
  root.innerHTML = html;
}

if (toolOutput) render(toolOutput);
</script>
</body>
</html>
""",
)


CHART_TEMPLATE = WidgetTemplate(
    uri="widget://chart",
    name="Chart",
    description="Data visualization charts",
    html="""
<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg-color: #ffffff;
  --text-color: #1a1a1a;
}
[data-theme="dark"] {
  --bg-color: #1a1a1a;
  --text-color: #ffffff;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg-color);
  color: var(--text-color);
  padding: 16px;
}
.chart-container {
  max-width: 800px;
  margin: 0 auto;
  position: relative;
}
.chart-title {
  font-size: 18px;
  font-weight: 600;
  text-align: center;
  margin-bottom: 8px;
}
.chart-subtitle {
  font-size: 14px;
  opacity: 0.7;
  text-align: center;
  margin-bottom: 16px;
}
canvas { max-width: 100%; }
</style>
</head>
<body>
<div id="root"></div>
<script>
const { toolOutput, theme } = window.openai || {};
document.body.dataset.theme = theme || 'light';

const defaultColors = [
  '#2196f3', '#4caf50', '#ff9800', '#f44336', '#9c27b0',
  '#00bcd4', '#8bc34a', '#ffc107', '#e91e63', '#673ab7'
];

function render(data) {
  const root = document.getElementById('root');

  let html = '<div class="chart-container">';
  if (data.title) html += '<div class="chart-title">' + data.title + '</div>';
  if (data.subtitle) html += '<div class="chart-subtitle">' + data.subtitle + '</div>';
  html += '<canvas id="chart"></canvas>';
  html += '</div>';
  root.innerHTML = html;

  const ctx = document.getElementById('chart').getContext('2d');

  const datasets = (data.data?.datasets || []).map((ds, i) => ({
    label: ds.label,
    data: ds.data,
    backgroundColor: ds.backgroundColor || ds.color || defaultColors[i % defaultColors.length] + '40',
    borderColor: ds.borderColor || ds.color || defaultColors[i % defaultColors.length],
    fill: ds.fill || false,
    tension: ds.tension || 0.1,
  }));

  new Chart(ctx, {
    type: data.chartType || 'line',
    data: {
      labels: data.data?.labels || [],
      datasets: datasets
    },
    options: {
      responsive: data.options?.responsive !== false,
      maintainAspectRatio: data.options?.maintainAspectRatio !== false,
      plugins: data.options?.plugins || {},
      scales: data.options?.scales || {}
    }
  });
}

if (toolOutput) render(toolOutput);
</script>
</body>
</html>
""",
)


# Template registry
WIDGET_TEMPLATES: dict[str, WidgetTemplate] = {
    "list-view": LIST_VIEW_TEMPLATE,
    "carousel": CAROUSEL_TEMPLATE,
    "form": FORM_TEMPLATE,
    "status": STATUS_TEMPLATE,
    "chart": CHART_TEMPLATE,
}


def get_template(name: str) -> WidgetTemplate | None:
    """Get a widget template by name.

    Args:
        name: Template name (e.g., "list-view", "carousel").

    Returns:
        WidgetTemplate or None if not found.
    """
    return WIDGET_TEMPLATES.get(name)


def register_template(name: str, template: WidgetTemplate) -> None:
    """Register a custom widget template.

    Args:
        name: Template name.
        template: Widget template.
    """
    WIDGET_TEMPLATES[name] = template
