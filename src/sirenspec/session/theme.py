"""Brand theme, colour-mode resolution, and Crest banner rendering for ``sirenspec launch``.

Every colour the studio uses is defined here once and reused — no widget hard-codes a hex
value.  The palette derives from ``docs/docs.json`` (primary ``#0D9373`` / light ``#07C983``)
and the converged terminal design in ``tui-final-design.html``.

The Crest banner is generated pixel-for-pixel from the ``<rect>`` geometry of
``docs/logo/crest.svg`` on a 28×28 grid and rendered with Unicode half-blocks (``▀``) so each
terminal row carries two pixel rows.  ``NO_COLOR`` / ``--plain`` fall back to a monochrome
silhouette.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from rich.text import Text
from textual.theme import Theme

# ---------------------------------------------------------------------------
# Brand palette — single source of truth (mirrors tui-final-design.html).
# ---------------------------------------------------------------------------

PRIMARY = "#0D9373"
LIGHT = "#07C983"
CREST_DARK = "#0a6b6b"
CREST_LIGHT = "#3dd6c8"

TERM_BG = "#0c1412"
TERM_BG_2 = "#0f1a18"
TERM_BORDER = "#1f3a35"
TERM_FG = "#c9d6d2"
TERM_DIM = "#6f8a83"
TERM_MUTED = "#54716b"
TERM_FAINT = "#46615b"
YOU = "#e6b450"

# Eye pixel colour — near-black so it reads against the body on dark terminals.
CREST_EYE = "#0a0e10"

# Plain-mode foreground used when colour is disabled (monochrome silhouette + chrome).
PLAIN_FG = "default"

CREST_GRID = 28


@dataclass(frozen=True)
class CrestRect:
    """A single filled rectangle of the Crest sprite on the 28×28 pixel grid.

    Mirrors one ``[col, row, colSpan, rowSpan, color]`` entry from the reference renderer
    in ``tui-final-design.html``, which is itself derived from ``docs/logo/crest.svg``
    (4px source units; ``col = (x-44)/4``, ``row = (y-11)/4``).

    :param col: Left pixel column (0-based) on the 28-wide grid.
    :param row: Top pixel row (0-based) on the 28-tall grid.
    :param col_span: Width of the rectangle in pixels.
    :param row_span: Height of the rectangle in pixels.
    :param color: Fill colour hex string.
    """

    col: int
    row: int
    col_span: int
    row_span: int
    color: str


# Exact rect geometry of the Crest, transcribed from tui-final-design.html.
CREST_RECTS: tuple[CrestRect, ...] = (
    # Wings (left)
    CrestRect(0, 6, 4, 2, CREST_DARK),
    CrestRect(2, 8, 2, 2, CREST_DARK),
    CrestRect(0, 10, 6, 2, CREST_LIGHT),
    CrestRect(2, 12, 4, 2, CREST_LIGHT),
    # Wings (right)
    CrestRect(24, 6, 4, 2, CREST_DARK),
    CrestRect(24, 8, 2, 2, CREST_DARK),
    CrestRect(22, 10, 6, 2, CREST_LIGHT),
    CrestRect(22, 12, 4, 2, CREST_LIGHT),
    # Body
    CrestRect(8, 6, 12, 2, CREST_LIGHT),
    CrestRect(8, 8, 12, 2, CREST_LIGHT),
    CrestRect(8, 10, 12, 2, CREST_LIGHT),
    CrestRect(8, 12, 12, 2, CREST_LIGHT),
    # Eyes
    CrestRect(10, 8, 2, 2, CREST_EYE),
    CrestRect(16, 8, 2, 2, CREST_EYE),
    # Tail
    CrestRect(10, 14, 8, 2, CREST_DARK),
    CrestRect(11, 16, 6, 2, CREST_DARK),
    CrestRect(12, 18, 4, 2, CREST_DARK),
    CrestRect(13, 20, 2, 2, CREST_DARK),
)


@dataclass(frozen=True)
class ColorMode:
    """Resolved colour preference for a launch session.

    :param enabled: ``True`` when ANSI colour should be emitted, ``False`` for the
        monochrome ``--plain`` / ``NO_COLOR`` / non-TTY path.
    """

    enabled: bool


def resolve_color_mode(plain: bool = False, *, is_tty: bool = True) -> ColorMode:
    """Decide whether colour is enabled for this session.

    Colour is disabled when the caller passes ``--plain``, when the ``NO_COLOR``
    environment variable is set (any value, per the ``no-color.org`` convention), or
    when stdout is not an interactive terminal.

    :param plain: ``True`` when the ``--plain`` flag was supplied.
    :param is_tty: Whether stdout is an interactive terminal.
    :returns: The resolved :class:`ColorMode`.
    """
    if plain or "NO_COLOR" in os.environ or not is_tty:
        return ColorMode(enabled=False)
    return ColorMode(enabled=True)


def build_crest_pixels() -> dict[tuple[int, int], str]:
    """Expand :data:`CREST_RECTS` into a ``(row, col) -> colour`` pixel map.

    :returns: Mapping of each filled pixel coordinate to its colour hex string.
    """
    pixels: dict[tuple[int, int], str] = {}
    for rect in CREST_RECTS:
        for dr in range(rect.row_span):
            for dc in range(rect.col_span):
                pixels[(rect.row + dr, rect.col + dc)] = rect.color
    return pixels


def render_crest(color_mode: ColorMode) -> Text:
    """Render the Crest mascot as a Rich :class:`~rich.text.Text` block.

    In colour mode each terminal row uses the upper-half-block ``▀`` so two pixel rows
    map onto one text row: the glyph foreground paints the top pixel and the glyph
    background paints the bottom pixel.  In plain mode a monochrome ``█``/space
    silhouette is produced instead.

    :param color_mode: The resolved colour mode for the session.
    :returns: A 14-row Rich ``Text`` sprite (28 columns wide).
    """
    pixels = build_crest_pixels()
    if not color_mode.enabled:
        return render_crest_plain(pixels)
    return render_crest_color(pixels)


def render_crest_color(pixels: dict[tuple[int, int], str]) -> Text:
    """Render the half-block colour sprite from a pixel map.

    :param pixels: ``(row, col) -> colour`` map from :func:`build_crest_pixels`.
    :returns: A Rich ``Text`` with one ``▀`` per cell, foreground = top pixel,
        background = bottom pixel.
    """
    text = Text(no_wrap=True)
    for text_row in range(CREST_GRID // 2):
        top_row = text_row * 2
        bottom_row = top_row + 1
        for col in range(CREST_GRID):
            top = pixels.get((top_row, col))
            bottom = pixels.get((bottom_row, col))
            if top is None and bottom is None:
                text.append(" ")
                continue
            fg = top if top is not None else TERM_BG
            bg = bottom if bottom is not None else TERM_BG
            text.append("▀", style=f"{fg} on {bg}")
        if text_row < CREST_GRID // 2 - 1:
            text.append("\n")
    return text


def render_crest_plain(pixels: dict[tuple[int, int], str]) -> Text:
    """Render a monochrome silhouette of the Crest for ``--plain`` / ``NO_COLOR``.

    :param pixels: ``(row, col) -> colour`` map from :func:`build_crest_pixels`.
    :returns: A Rich ``Text`` silhouette using ``▀``/``▄``/``█``/space, no colour.
    """
    text = Text(no_wrap=True)
    for text_row in range(CREST_GRID // 2):
        top_row = text_row * 2
        bottom_row = top_row + 1
        for col in range(CREST_GRID):
            top = (top_row, col) in pixels
            bottom = (bottom_row, col) in pixels
            if top and bottom:
                text.append("█")
            elif top:
                text.append("▀")
            elif bottom:
                text.append("▄")
            else:
                text.append(" ")
        if text_row < CREST_GRID // 2 - 1:
            text.append("\n")
    return text


def build_theme(color_mode: ColorMode) -> Theme:
    """Build the Textual :class:`~textual.theme.Theme` for the studio.

    The theme exposes the brand palette through Textual design tokens (``$primary``,
    ``$secondary``, ``$surface`` …) so the app's TCSS never hard-codes a hex value.
    In plain mode a desaturated dark theme is returned so colour-derived contrast still
    works while ANSI colour output is suppressed by the console layer.

    :param color_mode: The resolved colour mode for the session.
    :returns: A registered-ready :class:`~textual.theme.Theme` named ``"sirenspec"``.
    """
    return Theme(
        name="sirenspec",
        primary=PRIMARY,
        secondary=LIGHT,
        accent=CREST_LIGHT,
        foreground=TERM_FG,
        background=TERM_BG,
        surface=TERM_BG_2,
        panel=TERM_BG_2,
        success=LIGHT,
        warning=YOU,
        error="#ff5f57",
        dark=True,
        variables={
            "border": TERM_BORDER,
            "text-muted": TERM_DIM,
            "text-disabled": TERM_FAINT,
        },
    )
